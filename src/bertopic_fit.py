from pathlib import Path
from logging import Logger
from typing import Union, List, Tuple
import torch
from torch import Tensor
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
# from sklearn.cluster import HDBSCAN # Run this version if you don't have GPU
# from umap import UMAP # Run this version if you don't have GPU
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import logger, get_json

# FIts a single model. Only setup for HDBSCAN

class BertTopicFit:
    """
    Performs Latent Dirichlet Allocation (LDA) model on selected parameters.

    Attributes
    ----------
    params_json : Path
        JSON file path with model parameters.
    stop_words_path : Path
        Path to stop words txt.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        params_json: Path,
        stop_words_path: Path,
        logger: Logger = logger,
    ):

        self.params_json = get_json(params_json)
        self.stop_words_path = stop_words_path
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.logger = logger

    def fit(
        self,
        docs: List[str],
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
    ) -> Tuple[BERTopic, List[str], np.ndarray]:
        """
        Fits Bertopic.

        Parameters
        ----------
        Docs : List[str]
            Docs.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.

        Returns
        -------
        topic_model: BERTopic
            BERTopic fit model.
        """

        # Embedding and topn
        embedding_model = self.params_json["embedding_model"]
        topn = self.params_json["topn"]

        # Model parameters
        model_params = self.params_json["params"]
        min_topic_size = model_params["min_topic_size"]
        umap_n_components = model_params["umap_n_components"]
        umap_n_neighbors = model_params["umap_n_neighbors"]
        umap_min_dist = model_params["umap_min_dist"]
        hdbscan_min_samples = model_params["hdbscan_min_samples"]

        # Models
        umap_model = UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=hdbscan_min_samples,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=topn,
            embedding_model=embedding_model,
            language="brazilian portuguese",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
        )

        topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)

        return topic_model, topics, probs
