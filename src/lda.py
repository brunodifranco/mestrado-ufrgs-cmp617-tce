import os
import pickle
from pathlib import Path
from logging import Logger
from typing import List, Tuple
from tqdm import tqdm
import spacy
import nltk
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary
from data_cleaning import DataCleaning
from utils.utils import logger, get_json
from utils.nlp import preprocess, remove_stop_words, stemmer_pt, lemma_pt


class LDASingleModel:
    """
    Performs Latent Dirichlet Allocation (LDA) model on selected parameters.

    Attributes
    ----------
    params_json : Path
        JSON file path with model parameters.
    model_name : str
        LDA model name to be saved.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        params_json: Path,
        model_name: str,
        logger: Logger = logger,
    ):

        self.params_json = get_json(params_json)
        self.model_name = model_name
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

        nlp_normalization_method = self.params_json["nlp_normalization_method"]

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

        if nlp_normalization_method == "stemmer":
            self.logger.info("Running stemmer")

            tqdm.pandas()
            df["DS_OBJETO_NLP"] = df["DS_OBJETO_NLP"].progress_apply(
                stemmer_pt
            )  # Applies stemming

        elif nlp_normalization_method == "lemmatization":
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

    def create_corpus(self, vec: List[str]) -> Tuple[List, Dictionary]:
        """
        Creates corpus to be used in LDA (Latent Dirichlet Allocation) model.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.

        Returns
        -------
        corpus, id2word: Tuple[List, Dictionary]
            Corpus and gensim Dictionary.
        """

        n_filter = self.params_json["filter"]

        id2word = Dictionary(vec)
        tokens_to_remove = [
            token for token, freq in id2word.dfs.items() if freq < n_filter
        ]
        id2word.filter_tokens(bad_ids=tokens_to_remove)
        id2word.compactify()
        corpus = [id2word.doc2bow(text) for text in vec]

        self.logger.info("Corpus created!")

        return corpus, id2word

    def fit(self, corpus: List, id2word: Dictionary) -> LdaMulticore:
        """
        Fits LDA (Latent Dirichlet Allocation) model.

        Parameters
        ----------
        corpus : List
            Corpus.
        id2word : Dictionary
            Gensim Dictionary.

        Returns
        -------
        lda_model: LdaMulticore
            LDA (Latent Dirichlet Allocation) fit model.
        """

        model_params = self.params_json["params"]
        model_params["corpus"] = corpus
        model_params["id2word"] = id2word
        model_params["per_word_topics"] = True
        model_params["random_state"] = 42

        lda_model = LdaMulticore(**model_params)

        return lda_model

    def save(
        self,
        lda_model: LdaMulticore,
        vec: List[str],
        corpus: List,
        id2word: Dictionary,
    ):
        """
        Saves LDA model in pickle format, as well as its corpus, vec and id2word.

        Parameters
        ----------
        lda_model: LdaMulticore
            LDA (Latent Dirichlet Allocation) fit model.
        vec : List[str]
            List of text data for model optimization.
        corpus : List
            Corpus.
        id2word : Dictionary
            Gensim Dictionary.
        """

        path = f"models/lda/{self.model_name}/"

        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(lda_model, open(path + "model.pkl", "wb"))
        pickle.dump(vec, open(path + "vec.pkl", "wb"))
        pickle.dump(corpus, open(path + "corpus.pkl", "wb"))
        pickle.dump(id2word, open(path + "id2word.pkl", "wb"))

        self.logger.info(f"Model and its files saved to {path}")

    def run(self):
        """Runs the pipeline"""

        self.logger.setLevel("INFO")
        vec = self.nlp_preprocessing()
        corpus, id2word = self.create_corpus(vec)

        self.logger.setLevel("WARNING")
        lda_model = self.fit(corpus, id2word)

        self.logger.setLevel("INFO")
        self.save(lda_model, vec, corpus, id2word)


if __name__ == "__main__":
    lda = LDASingleModel(
        params_json="src/opt_outputs/results_stemmer_with_filter_500.json",
        model_name="model_test",
    )
    lda.run()
