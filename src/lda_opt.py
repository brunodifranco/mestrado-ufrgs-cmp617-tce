import os
import json
from typing import Union, List, Dict
from logging import Logger
import optuna
from tqdm import tqdm
import spacy
import nltk
from gensim.models import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from data_cleaning import DataCleaning
from utils.utils import logger
from utils.nlp import preprocess, remove_stop_words, stemmer_pt, lemma_pt


class LDAOptimization:
    """
    Performs Bayesian Optimization on Latent Dirichlet Allocation (LDA) model.

    Attributes
    ----------
    nlp_normalization_method : str
        NLP normalization method to choose: either 'stemmer' or 'lemmatization'
    n_trials : int
        Number of trials in optimization.
    topn : int
        Number of top words to be extracted from each topic. 
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        nlp_normalization_method: str,
        n_trials: int,
        topn: int,
        logger: Logger = logger,
    ):

        self.nlp_normalization_method = nlp_normalization_method
        self.n_trials = n_trials
        self.topn = topn
        self.logger = logger

    def nlp_preprocessing(self) -> List[str]:
        """
        Cleans data and performs NLP techniques.

        Returns
        -------
        vec : List[str]
            List of text data for model coherence calculation.
        """

        cleaning_pipeline = DataCleaning()
        df = cleaning_pipeline.run()

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
            )  # Applies lemmatization

        else:
            self.logger.error("TypeError")
            raise TypeError(
                "Please choose either 'stemmer' or 'lemmatization' as the nlp_normalization_method"
            )

        vec = df["DS_OBJETO_NLP"].values.tolist()
        return vec

    def bayesian_opt_objective(self, trial: optuna.Trial, vec: List[str]) -> float:
        """
        Objective function for Bayesian Optimization to tune parameters for LDA (Latent Dirichlet Allocation) model,
        returning the Coherence score c_v.

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
        coherence_cv: float
            Coherence score c_v of the LDA model.
        """

        n_filter = trial.suggest_int("n_filter", 0, 500, step=50)
        num_topics = trial.suggest_int("num_topics", 5, 10, step=1)
        chunksize = trial.suggest_int("chunksize", 80, 180, step=10)
        passes = trial.suggest_int("passes", 5, 20, step=1)
        alpha = trial.suggest_float("alpha", 0.01, 1, step=0.01)
        eta = trial.suggest_float("eta", 0.01, 0.91, step=0.01)
        decay = trial.suggest_float("decay", 0.5, 1, step=0.1)

        # Create corpus
        id2word = Dictionary(vec)
        tokens_to_remove = [
            token for token, freq in id2word.dfs.items() if freq < n_filter
        ]
        id2word.filter_tokens(bad_ids=tokens_to_remove)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in vec]

        lda_model = LdaMulticore(
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
        coherence_model_cv = CoherenceModel(
            model=lda_model, texts=vec, dictionary=id2word, coherence="c_v", topn=self.topn
        )
        coherence_cv = coherence_model_cv.get_coherence()

        return coherence_cv

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

        # Optimizer
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.bayesian_opt_objective(trial, vec),
            n_trials=self.n_trials,
        )
        trial = study.best_trial  # Get best trial

        # Store results
        results = {}
        results["nlp_normalization_method"] = self.nlp_normalization_method
        results["topn"] = self.topn
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

        output_dir = "src/lda_opt_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = f"{output_dir}/results_{self.nlp_normalization_method}_topn_{self.topn}.json"
        with open(output_path, "w") as json_file:
            json.dump(results, json_file)

        self.logger.info("Results saved to JSON")

    def run(self):
        """Runs the optimizer"""

        self.logger.setLevel("INFO")
        vec = self.nlp_preprocessing()

        self.logger.setLevel("WARNING")
        results = self.get_opt(vec)

        self.logger.setLevel("INFO")
        self.save_results(results)

        self.logger.info("Optimizer completed!")


if __name__ == "__main__":
    optimizer = LDAOptimization(
        nlp_normalization_method="stemmer",  # Method to choose: either stemmer or lemmatization
        topn=5, # Number of top words to be extracted from each topic.
        n_trials=30,  # Number of trials for optimization
    )
    optimizer.run()
