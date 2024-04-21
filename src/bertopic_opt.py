import os
import json
from pathlib import Path
from logging import Logger
from typing import Union, List, Dict, Tuple
import torch
from torch import Tensor
import numpy as np
import optuna
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from cuml.preprocessing import normalize
from evaluate_bertopic import load_inputs
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from utils.utils import logger
from data_cleaning import DataCleaning


class BERTopicOptimization:
    """
    Performs Bayesian Optimization on BERTopic model.

    Attributes
    ----------
    embedding_model : str
        Sentence Transformer model name or path.
    topn : int
        Number of top words to be extracted from each topic.
    n_trials : int
        Number of trials in optimization.
    stop_words_path : Path
        Stop words path.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        embedding_model: str,
        topn: int,
        n_trials: int,
        stop_words_path: Path,
        logger: Logger = logger,
    ):

        self.embedding_model = embedding_model
        self.topn = topn
        self.n_trials = n_trials
        self.stop_words_path = stop_words_path
        self.logger = logger
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def bayesian_opt_objective(
        self,
        trial: optuna.Trial,
        docs: List[str],
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
    ) -> float:
        """
        Objective function for Bayesian Optimization to tune parameters for BERTopic model,
        returning the Coherence score c_v.

        Parameters
        ----------
        trial : optuna.Trial
            A single trial of an optimization experiment. The objective function uses this to suggest new parameters.
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.

        Returns
        -------
        coherence_cv: float
            Coherence score c_v of the BERTopic model.
        """

        # Suggested Params
        # min_topic_size = trial.suggest_int("min_topic_size", 20, 50, step=5)
        k_means_clusters = trial.suggest_int("k_means_clusters", 30, 200, step=5)
        k_means_n_init = trial.suggest_int("k_means_n_init", 5, 20, step=1)
        k_means_max_iter = trial.suggest_int("k_means_max_iter", 200, 500, step=10)
        # umaap_dist_metric = trial.suggest_categorical(
        #     "k_means_algorithm",
        #     [
        #         "manhattan",
        #         "euclidean",
        #         "chebyshev",
        #         "canberra",
        #         "sqeuclidean",
        #         "cosine",
        #     ],
        # )
        umap_n_components = trial.suggest_int("umap_n_components", 2, 12, step=1)
        umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 10, 40, step=1)
        umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 0.9, step=0.05)
        # hdbscan_min_samples = trial.suggest_int("hdbscan_min_samples", 5, 20, step=1)

        # nr_topics = trial.suggest_int("nr_topics", 5, 10, step=1)

        # Models
        umap_model = UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="manhattan",
            random_state=42,
        )
        # hdbscan_model = HDBSCAN(
        #     min_cluster_size=min_topic_size,
        #     min_samples=hdbscan_min_samples,
        #     gen_min_span_tree=True,
        #     prediction_data=True,
        # )

        # from cuml.cluster import KMeans
        from sklearn.cluster import KMeans

        cluster_model = KMeans(
            n_clusters=k_means_clusters,
            n_init=k_means_n_init,
            max_iter=k_means_max_iter,
            random_state=42,
        )

        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=self.topn,
            embedding_model=self.embedding_model,
            language="brazilian portuguese",
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            representation_model=KeyBERTInspired(),
        )
        topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)

        # Evaluating Results
        cleaned_docs = topic_model._preprocess_text(docs)
        analyzer = vectorizer_model.build_analyzer()
        tokens = [analyzer(doc) for doc in cleaned_docs]

        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topics = topic_model.get_topics()
        topics.pop(-1, None)  # removes the outliers topics

        # topics = {
        #     topic_id: terms
        #     for topic_id, terms in topics.items()
        #     if all(term != "" for term, weight in terms)
        # }
        # TESTAR RODAR AQUI SEM REMOVER O NEGOCIO MSM

        topic_words = [
            [word for word, _ in topic_model.get_topic(topic) if word != ""]
            for topic in topics
        ]

        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_cv = coherence_model.get_coherence()

        return coherence_cv

    def get_opt(
        self,
        docs: List[str],
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
    ) -> Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]:
        """
        Perform optimization of BERTopic model parameters using Bayesian Optimization.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.
        embeddings : Union[List[Tensor], np.ndarray, Tensor]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.

        Returns
        -------
        Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]
        """

        # Optimizer
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.bayesian_opt_objective(
                trial, docs, embeddings, vectorizer_model
            ),
            n_trials=self.n_trials,
        )
        trial = study.best_trial  # Get best trial

        # Store results
        results = {}
        results["embedding_model"] = self.embedding_model
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

        output_dir = "src/bertopic_opt_outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir + f"/{self.embedding_model}")

        output_path = (
            f"{output_dir}/{self.embedding_model}/results_topn_novo_{self.topn}_.json"
        )
        with open(output_path, "w") as json_file:
            json.dump(results, json_file)

        self.logger.info("Results saved to JSON")

    def run(self):
        """Runs the optimizer"""

        self.logger.setLevel("INFO")
        embeddings, vectorizer_model, docs = load_inputs("models/bertopic/inputs")

        self.logger.setLevel("WARNING")
        results = self.get_opt(docs, embeddings, vectorizer_model)

        self.logger.setLevel("INFO")
        self.save_results(results)

        self.logger.info("Optimizer completed!")


if __name__ == "__main__":
    optimizer = BERTopicOptimization(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Sentence Transformer model name or path
        topn=7,
        n_trials=70,  # Number of trials for optimization
        stop_words_path="src/utils/stop_words.txt",
    )
    optimizer.run()
