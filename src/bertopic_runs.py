import os
import json
import pickle
from pathlib import Path
from logging import Logger
from typing import Union, List, Tuple
import numpy as np
import torch
from torch import Tensor
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import logger, get_json
from evaluate_bertopic import coherence_score
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BerTopicRuns:
    """
    Runs the BerTopic model 20 times, to get the mean and std of the metrics
    c_uci, c_npmi, c_v, u_mass. It's important because of the model variability, due to UMAP.

    Attributes
    ----------
    inputs_path : Path
        Path to model inputs.
    params_json : Path
        JSON file path with model parameters.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        inputs_path: Path,
        params_json: Path,
        logger: Logger = logger,
    ):

        self.inputs_path = inputs_path
        self.params = get_json(params_json)
        self.logger = logger
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def load_model_inputs(
        self,
    ) -> Tuple[Union[List[Tensor], np.ndarray, Tensor], CountVectorizer, List[str]]:
        """
        Loads the model inputs: embeddings, vectorizer_model, docs

        Returns
        -------
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.
        """

        with open(self.inputs_path + "/embeddings.pkl", "rb") as file:
            embeddings = pickle.load(file)
        with open(self.inputs_path + "/vectorizer_model.pkl", "rb") as file:
            vectorizer_model = pickle.load(file)
        with open(self.inputs_path + "/docs.pkl", "rb") as file:
            docs = pickle.load(file)

        return embeddings, vectorizer_model, docs

    def run_models(
        self,
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
        docs: List[str],
    ):
        """
        Runs model.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.
        """

        num_runs = 100
        metrics = []

        best_cv_score = float("-inf")  # Initially set to -inf
        best_model = None
        best_topics = None
        best_probs = None

        for i in range(num_runs):

            print(f"Now running run number {i}")

            embedding_model = self.params["embedding_model"]
            topn = self.params["topn"]
            k_means_clusters = self.params["params"]["k_means_clusters"]
            k_means_n_init = self.params["params"]["k_means_n_init"]
            k_means_max_iter = self.params["params"]["k_means_max_iter"]
            umap_n_components = self.params["params"]["umap_n_components"]
            umap_n_neighbors = self.params["params"]["umap_n_neighbors"]
            umap_min_dist = self.params["params"]["umap_min_dist"]

            # Models
            umap_model = UMAP(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=42,
            )

            cluster_model = KMeans(n_clusters=k_means_clusters, 
                                    n_init=k_means_n_init,
                                    max_iter=k_means_max_iter,
                                    random_state=42)
            
            topic_model = BERTopic(
                nr_topics="auto",
                top_n_words=topn,
                embedding_model=embedding_model,
                language="brazilian portuguese",
                umap_model=umap_model,
                hdbscan_model=cluster_model,
                vectorizer_model=vectorizer_model,
                calculate_probabilities=True,
                representation_model=KeyBERTInspired(),
            )

            topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)

            c_uci = coherence_score(docs, topic_model, vectorizer_model, "c_uci")
            c_npmi = coherence_score(docs, topic_model, vectorizer_model, "c_npmi")
            c_v = coherence_score(docs, topic_model, vectorizer_model, "c_v")
            u_mass = coherence_score(docs, topic_model, vectorizer_model, "u_mass")

            self.logger.info(f"c_v = {c_v}")

            # Gets the best model based on "c_v"
            if c_v > best_cv_score:
                best_cv_score = c_v
                best_model = topic_model
                best_topics = topics
                best_probs = probs

            # Saving the metrics
            metrics.append(
                {
                    "run": i + 1,
                    "c_uci": c_uci,
                    "c_npmi": c_npmi,
                    "c_v": c_v,
                    "u_mass": u_mass,
                }
            )

        results = {"metrics": metrics}

        # Saves the best model
        path = f"models/bertopic/best_model/"
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(best_model, open(path + "model.pkl", "wb"))
        pickle.dump(best_topics, open(path + "topics.pkl", "wb"))
        pickle.dump(best_probs, open(path + "probs.pkl", "wb"))

        self.logger.info(f"Best model and its files saved to {path}")

        # Saves results
        path_metrics = f"models/bertopic/best_model/metrics"
        if not os.path.exists(path_metrics):
            os.makedirs(path_metrics)

        with open(path_metrics + "/bertopic_metrics.json", "w") as json_file:
            json.dump(results, json_file, indent=4, sort_keys=True)

        self.logger.info("Best metrics saved, with ")

    def run(self):
        """Runs the pipeline"""

        self.logger.setLevel("INFO")
        embeddings, vectorizer_model, docs = self.load_model_inputs()

        self.logger.setLevel("WARNING")
        self.run_models(embeddings, vectorizer_model, docs)


if __name__ == "__main__":
    lda = BerTopicRuns(
        inputs_path="models/bertopic/inputs",
        params_json="src/bertopic_opt_outputs/paraphrase-multilingual-MiniLM-L12-v2/results_topn_novo_7_.json",
    )
    lda.run()
