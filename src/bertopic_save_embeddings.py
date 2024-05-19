import os
from pathlib import Path
from logging import Logger
from typing import Union, List, Tuple
import torch
import pickle
from torch import Tensor
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from cuml.preprocessing import normalize
# from sklearn.preprocessing import normalize # Run this version if you don't have GPU
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import logger
from data_cleaning import DataCleaning


class BERTopicEmbeddings:
    """
    Creates and saves BERTopic embeddings, vectorizer_model and docs.

    Attributes
    ----------
    embedding_model : str
        Sentence Transformer model name or path.
    stop_words_path : Path
        Stop words path.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        embedding_model: str,
        stop_words_path: Path,
        logger: Logger = logger,
    ):

        self.embedding_model = embedding_model
        self.stop_words_path = stop_words_path
        self.logger = logger
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def pre_process_bert(
        self,
    ) -> Tuple[Union[List[Tensor], np.ndarray, Tensor], CountVectorizer, List[str]]:
        """
        Pre process to use in Bertopic, returning the documents, embeddings and vectorizer model.

        Returns
        -------
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.
        """

        data_cleaning_pipeline = DataCleaning()
        df = data_cleaning_pipeline.run()

        self.logger.info("Running NLP treatment")

        data = Dataset.from_pandas(df)
        docs = data["DS_OBJETO"]

        stop_words = []
        with open(self.stop_words_path, "r") as file:
            for row in file:
                stop_words.append(row.strip())

        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            strip_accents="unicode",
            token_pattern=r'(?u)\b[A-Za-zÀ-ÿ]{4,}\b'  # Regex removing numbers and also tokens lower than 4 characters
        )
        # Filtering docs that are completely null after stopwords removal
        vec = vectorizer_model.fit_transform(docs)
        doc_lengths = vec.sum(axis=1)
        non_empty_docs = doc_lengths > 0

        df_filtered = df[non_empty_docs]
        data = Dataset.from_pandas(df_filtered)
        docs = data["DS_OBJETO"]

        sentence_model = SentenceTransformer(
            model_name_or_path=self.embedding_model, device=self.device
        )
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        embeddings = normalize(embeddings)

        return embeddings, vectorizer_model, docs

    def save_embeddings(
        self,
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
        docs: List[str],
    ):
        """
        Saves documents, embeddings and vectorizer model.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.
        """

        path = f"models/bertopic/inputs/"
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(embeddings, open(path + "embeddings.pkl", "wb"))
        pickle.dump(vectorizer_model, open(path + "vectorizer_model.pkl", "wb"))
        pickle.dump(docs, open(path + "docs.pkl", "wb"))

        self.logger.info(f"BerTopic inputs saved to {path}")

    def run(self):
        """Runs the process"""

        embeddings, vectorizer_model, docs = self.pre_process_bert()
        self.save_embeddings(embeddings, vectorizer_model, docs)


if __name__ == "__main__":
    bertopic_inputs = BERTopicEmbeddings(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Sentence Transformer model name or path
        stop_words_path="src/utils/stop_words_bertopic.txt",
    )
    bertopic_inputs.run()
