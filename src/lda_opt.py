import json
from typing import Union, List, Dict
from logging import Logger
import optuna
from tqdm import tqdm
import nltk
from gensim.models import CoherenceModel, ldamulticore
from gensim.corpora import Dictionary
from data_cleaning import DataCleaning
from utils.utils import logger
from utils.nlp import preprocess, remove_stop_words, stemmer_pt, lemma_pt
import spacy




class LDAOptimization:
    """
    Performs Bayesian Optimization on Latent Dirichlet Allocation (LDA) model.

    Attributes
    ----------
    nlp_normalization_method : str
        NLP normalization method to choose: either 'stemmer' or 'lemmatization'
    n_filter : int
        Minimum frequency to retain a token in the dictionary.
    n_trials : int
        Number of trials in optimization.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        nlp_normalization_method: str,
        n_filter: int,
        n_trials: int,
        logger: Logger = logger,
    ):

        self.nlp_normalization_method = nlp_normalization_method
        self.n_filter = n_filter
        self.n_trials = n_trials
        self.logger = logger

    def nlp_preprocessing(self) -> List[str]:
        cleaning_pipeline = DataCleaning()
        df = cleaning_pipeline.run()

        # TODO - REMOVER AQUI DEPOIS
        df = df[df["ANO_LICITACAO"] >= 2021]

        self.logger.info("Running NLP treatment")

        df = df.assign(
            DS_OBJETO_NLP=df["DS_OBJETO"]
            .apply(
                lambda x: nltk.word_tokenize(x.lower(), language="portuguese")
            )  # Tokenize
            .apply(lambda x: [preprocess(word) for word in x])  # Other preprocessing
            .apply(lambda x: list(filter(None, x)))  # Removes items with none
            .apply(remove_stop_words)  # Removes stop words
            .apply(
                lambda x: [word for word in x if "rs" not in word]
            )  # Remove tokens containing "rs" (which are cities)
        )

        if self.nlp_normalization_method == "stemmer":
            self.logger.info("Running stemmer")

            tqdm.pandas()
            df["DS_OBJETO_NLP"] = df["DS_OBJETO_NLP"].progress_apply(
                stemmer_pt
            )  # Applies stemming

        elif self.nlp_normalization_method == "lemmatization":
            self.logger.info("Running lemmatization")

            nlp = spacy.load("pt_core_news_md", disable=["parser", "tagger", "ner"])

            tqdm.pandas()
            df["DS_OBJETO_NLP"] = df["DS_OBJETO_NLP"].progress_apply(
                lambda x: lemma_pt(nlp, x)
            )  # Aplica a lematização


        else:
            self.logger.error("TypeError")
            raise TypeError(
                "Please choose either 'stemmer' or 'lemmatization' as the nlp_normalization_method"
            )

        vec = df["DS_OBJETO_NLP"].values.tolist()
        return vec

    def bayesian_opt_objective(
        self, trial: optuna.Trial, id2word: Dictionary, corpus: List, vec: List[str]
    ) -> float:
        """
        Objective function for Bayesian Optimization to tune parameters for LDA (Latent Dirichlet Allocation) model,
        returning the Coherence score c-v (Cosine Similarity).

        Parameters
        ----------
        trial : optuna.Trial
            A single trial of an optimization experiment. The objective function uses this to suggest new parameters.
        id2word : gensim.corpora.Dictionary
            Gensim dictionary mapping of word IDs to words.
        corpus : List
            List of Bag-of-Words corpus.
        vec : List[str]
            List of text data for model coherence calculation.

        Returns
        -------
        float
            Coherence score c-v (Cosine Similarity) of the LDA model.
        """

        num_topics = trial.suggest_int("num_topics", 5, 7, step=1)
        chunksize = trial.suggest_int("chunksize", 80, 180, step=10)
        passes = trial.suggest_int("passes", 5, 20, step=1)
        alpha = trial.suggest_float("alpha", 0.01, 1, step=0.01)
        eta = trial.suggest_float("eta", 0.01, 0.91, step=0.01)
        decay = trial.suggest_float("decay", 0.5, 1, step=0.1)

        lda_model = ldamulticore.LdaMulticore(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics,
            chunksize=chunksize,
            passes=passes,
            alpha=alpha,
            eta=eta,
            decay=decay,
            random_state=42,
            per_word_topics=True,
        )

        # Coherence Score
        coherence_model_lda = CoherenceModel(
            model=lda_model, texts=vec, dictionary=id2word, coherence="c_v"
        )
        coherence_lda = coherence_model_lda.get_coherence()

        return coherence_lda

    def get_opt(
        self, vec: List[str]
    ) -> Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]:
        """
        Perform optimization of LDA (Latent Dirichlet Allocation) model parameters using Bayesian Optimization.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.

        Returns
        -------
        Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]
        """

        # Create corpus
        id2word = Dictionary(vec)
        tokens_a_remover = [
            token for token, freq in id2word.dfs.items() if freq < self.n_filter
        ]
        id2word.filter_tokens(bad_ids=tokens_a_remover)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in vec]

        # Optimizer
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.bayesian_opt_objective(trial, id2word, corpus, vec),
            n_trials=self.n_trials,
        )
        trial = study.best_trial

        # Store results
        results = {}
        results["filter"] = self.n_filter
        results["nlp_normalization_method"] = self.nlp_normalization_method
        results["best_score"] = trial.value
        results["params"] = trial.params

        self.logger.info("Results stored")
        return results

    def save_results(
        self, results: Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]
    ):
        """
        Save results to JSON.

        Parameters
        ----------
        results: Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]
            Dict with results.
        """

        output_path = f"src/opt_outputs/results_{self.nlp_normalization_method}_with_filter_{self.n_filter}.json"
        with open(output_path, "w") as json_file:
            json.dump(results, json_file)

        self.logger.info("Results saved to JSON")

    def run(self):
        """
        Runs the optimizer
        """
        vec = self.nlp_preprocessing()
        results = self.get_opt(vec)
        self.save_results(results)

        self.logger.info("Optimizer completed!")


if __name__ == "__main__":
    optimizer = LDAOptimization(
        nlp_normalization_method="lemmatization",  # method to choose: either stemmer or lemmatization
        n_filter=500 ,
        n_trials=50,
    )
    optimizer.run()
