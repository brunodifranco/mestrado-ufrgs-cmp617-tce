import os
import json
from typing import Union, List, Dict
from logging import Logger
import optuna
from pathlib import Path
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from data_cleaning import DataCleaning
from utils.utils import logger
from gensim.models import CoherenceModel
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from cuml.preprocessing import normalize
import torch
from datasets import Dataset
from gensim.corpora import Dictionary


class BERTopicOptimization:
    """
    Performs Bayesian Optimization on BERTopic model.

    Attributes
    ----------
    embedding_model : str
        Sentence Transformer model name or path.
    n_trials : int
        Number of trials in optimization.
    logger : Logger, defaults to logger
        logger.
    """

    def __init__(
        self,
        embedding_model: str,
        n_trials: int,
        stop_words_path: Path,
        logger: Logger = logger,
    ):

        self.embedding_model = embedding_model
        self.n_trials = n_trials
        self.stop_words_path = stop_words_path
        self.logger = logger
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def nlp_preprocessing(self) -> List[str]:
        """
        Cleans data and performs NLP techniques.

        Returns
        -------
        vec : List[str]
            List of text data for model coherence calculation.
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
            stop_words=stop_words, strip_accents="unicode"
        )

        sentence_model = SentenceTransformer(
            model_name_or_path=self.embedding_model, device=self.device
        )
        embeddings = sentence_model.encode(docs, show_progress_bar=True)

        embeddings = normalize(embeddings)

        return embeddings, vectorizer_model, docs

    def bayesian_opt_objective(
        self,
        trial: optuna.Trial,
        docs: List[str],
        embeddings: List[float],
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
            List of text data.
        embeddings : List[float]
            List of precomputed embeddings.
        vectorizer_model : CountVectorizer
            CountVectorizer model.

        Returns
        -------
        coherence_cv: float
            Coherence score c_v of the BERTopic model.
        """
        
        # Suggested Params
        min_topic_size = trial.suggest_int("min_topic_size", 350, 800, step=50)
        umap_n_components = trial.suggest_int("umap_n_components", 2, 6, step=1)
        umap_n_neighbors = trial.suggest_int("umap_n_neighbors", 5, 20, step=1)
        umap_min_dist = trial.suggest_float("umap_min_dist", 0.0, 1.0, step=0.1)
        hdbscan_min_samples = trial.suggest_int("hdbscan_min_samples", 5, 20, step=1)

        # Models
        umap_model = UMAP(
            n_components=umap_n_components,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            min_samples=hdbscan_min_samples,
            gen_min_span_tree=True,
            prediction_data=True,
        )

        topic_model = BERTopic(
            nr_topics="auto",
            top_n_words=7,  # Fixed, so we can compare to LDA model
            embedding_model=self.embedding_model,
            language="brazilian portuguese",
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=KeyBERTInspired(),
        )
        topics, _ = topic_model.fit_transform(documents=docs, embeddings=embeddings)

        # Evaluating Results
        cleaned_docs = topic_model._preprocess_text(docs)
        analyzer = vectorizer_model.build_analyzer()
        tokens = [analyzer(doc) for doc in cleaned_docs]

        dictionary = Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        topics = topic_model.get_topics()
        topics.pop(-1, None)
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
        self, docs: List[str], embeddings: List[float], vectorizer_model: CountVectorizer
    ) -> Dict[str, Union[int, str, float, Dict[str, Union[int, float]]]]:
        """
        Perform optimization of BERTopic model parameters using Bayesian Optimization.

        Parameters
        ----------
        vec : List[str]
            List of text data for model optimization.
        embeddings : List[float]
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
            os.makedirs(output_dir)

        output_path = f"{output_dir}/results_{self.embedding_model}.json"
        with open(output_path, "w") as json_file:
            json.dump(results, json_file)

        self.logger.info("Results saved to JSON")

    def run(self):
        """Runs the optimizer"""

        self.logger.setLevel("INFO")
        embeddings, vectorizer_model, docs = self.nlp_preprocessing()

        self.logger.setLevel("WARNING")
        results = self.get_opt(docs, embeddings, vectorizer_model)

        self.logger.setLevel("INFO")
        self.save_results(results)

        self.logger.info("Optimizer completed!")


if __name__ == "__main__":
    optimizer = BERTopicOptimization(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Sentence Transformer model name or path
        n_trials=20,  # Number of trials for optimization
        stop_words_path="src/utils/stop_words.txt",
    )
    optimizer.run()
