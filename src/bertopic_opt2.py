import os
import pickle
from pathlib import Path
from logging import Logger
from typing import Union, List
import torch
from torch import Tensor
import numpy as np
import optuna
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from cuml.manifold import UMAP
from cuml.cluster import HDBSCAN, KMeans

# from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from utils.utils import logger
from evaluate_bertopic import load_inputs


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class BERTopicOptimization:
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
        self.best_cv_score = float("-inf")
        self.best_model_path = Path(f"models/bertopic/{embedding_model}/best_model/model.pkl")

    def bayesian_opt_objective(
        self,
        trial: optuna.Trial,
        docs: List[str],
        embeddings: Union[List[Tensor], np.ndarray, Tensor],
        vectorizer_model: CountVectorizer,
    ) -> float:
        # k_means_clusters = trial.suggest_int("k_means_clusters", 50, 200, step=5)
        # k_means_n_init = trial.suggest_int("k_means_n_init", 5, 20, step=1)
        # k_means_max_iter = trial.suggest_int("k_means_max_iter", 200, 500, step=10)
        alpha = trial.suggest_float("alpha", 0.5, 2, step=0.1)
        min_cluster_size = trial.suggest_int("min_cluster_size", 12, 40, step=2)
        max_cluster_size = trial.suggest_int(
            "max_cluster_size", 30000, 50000, step=1000
        )
        cluster_selection_method = trial.suggest_categorical(
            "cluster_selection_method", ["eom", "leaf"]
        )
        hdbscan_min_samples = trial.suggest_int("hdbscan_min_samples", 5, 25, step=1)
        umap_n_components = trial.suggest_int("umap_n_components", 2, 15, step=1)
        umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 10, 40, step=2)
        umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 0.9, step=0.05)
        nr_topics = trial.suggest_int("nr_topics", 12, 35, step=1)

        # cluster_model = KMeans(
        #         n_clusters=k_means_clusters,
        #         n_init=k_means_n_init,
        #         max_iter=k_means_max_iter,
        #         random_state=42,
        #     )

        cluster_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            min_samples=hdbscan_min_samples,
            alpha=alpha,
            cluster_selection_method=cluster_selection_method,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        topic_model = BERTopic(
            nr_topics=nr_topics,
            top_n_words=self.topn,
            embedding_model=self.embedding_model,
            language="brazilian portuguese",
            umap_model=UMAP(
                n_components=umap_n_components,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                random_state=42,
                # metric="cosine"
                metric="manhattan",
            ),
            hdbscan_model=cluster_model,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            representation_model=KeyBERTInspired(),
        )
        topics, probs = topic_model.fit_transform(documents=docs, embeddings=embeddings)
        coherence_cv = self.evaluate_model(docs, vectorizer_model, topic_model)

        # Check if the current model is better, save it if true
        if coherence_cv > self.best_cv_score:
            self.best_cv_score = coherence_cv
            self.save_model(topic_model)

        return coherence_cv

    def evaluate_model(self, docs, vectorizer_model, topic_model):

        # Evaluating Results
        cleaned_docs = topic_model._preprocess_text(docs)
        analyzer = vectorizer_model.build_analyzer()
        tokens = [analyzer(doc) for doc in cleaned_docs]

        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topics = topic_model.get_topics()
        topics.pop(-1, None)  # removes the outliers topics

        topics = {
            topic_id: terms
            for topic_id, terms in topics.items()
            if all(term != "" for term, weight in terms)
        }

        topic_words = [
            [word for word, _ in topic_model.get_topic(topic) if word != ""]
            for topic in topics
        ]

        topic_words = [sublist for sublist in topic_words if sublist]

        coherence_model = CoherenceModel(
            topics=topic_words,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_cv = coherence_model.get_coherence()

        return coherence_cv

    def save_model(self, model):
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.best_model_path, "wb") as f:
            pickle.dump(model, f)
        self.logger.info(f"New best model saved with c_v score: {self.best_cv_score}")

    def run(self):
        embeddings, vectorizer_model, docs = load_inputs("models/bertopic/inputs/LaBSE")
        self.optimize(docs, embeddings, vectorizer_model)

    def optimize(self, docs, embeddings, vectorizer_model):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self.bayesian_opt_objective(
                trial, docs, embeddings, vectorizer_model
            ),
            n_trials=self.n_trials,
        )


if __name__ == "__main__":
    optimizer = BERTopicOptimization(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        topn=7,
        n_trials=20,
        stop_words_path=Path("src/utils/stop_words_bertopic.txt"),
    )
    optimizer.run()
