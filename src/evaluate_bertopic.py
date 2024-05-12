import pickle
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
import numpy as np
from bertopic import BERTopic
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from cuml.cluster import HDBSCAN
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer
from utils.utils import logger


def load_inputs(
    inputs_path: Path,
) -> Tuple[Union[List[Tensor], np.ndarray, Tensor], CountVectorizer, List[str]]:
    """
    Loads the model inputs: embeddings, vectorizer_model, docs

    Parameters
    ----------
    inputs_path : Path
        Model inputs path.

    Returns
    -------
    embeddings : Union[List[Tensor], np.ndarray, Tensor]
        List of precomputed embeddings.
    vectorizer_model : CountVectorizer
        CountVectorizer model.
    docs : List[str]
        List of text data for model optimization.
    """

    with open(inputs_path + "/embeddings.pkl", "rb") as file:
        embeddings = pickle.load(file)
    with open(inputs_path + "/vectorizer_model.pkl", "rb") as file:
        vectorizer_model = pickle.load(file)
    with open(inputs_path + "/docs.pkl", "rb") as file:
        docs = pickle.load(file)

    return embeddings, vectorizer_model, docs


def load_model_outputs(
    model_path: Path,
) -> Tuple[BERTopic, List[str], np.ndarray]:
    """
    Loads the BERTopic model.

    Parameters
    ----------
    model_path : Path
        Model path.

    Returns
    -------
    topic_model, topics, probs : Tuple[BERTopic, List[str], np.ndarray]
        BerTopic model, topics and probabilities
    """

    with open(model_path + "/model.pkl", "rb") as file:
        topic_model = pickle.load(file)

    with open(model_path + "/topics.pkl", "rb") as file:
        topics = pickle.load(file)

    with open(model_path + "/probs.pkl", "rb") as file:
        probs = pickle.load(file)

    return topic_model, topics, probs


def coherence_score(
    docs: List[str],
    topic_model: BERTopic,
    vectorizer_model: CountVectorizer,
    coherence: str,
) -> float:
    """
    Calculates the desired coherence_score.

    Parameters
    ----------
    docs : List[str]
        Docs.
    topic_model : BERTopic
        BerTopic model.
    vectorizer_model : CountVectorizer
        CountVectorizer model.
    coherence : str
        Desired coherence metrics. Options are {"c_v", "c_npmi"}

    Returns
    --------
    coherence_score: float
        Coherence Score.
    """

    logger.setLevel("INFO")

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

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence=coherence,
    )
    coherence_score = coherence_model.get_coherence()

    return coherence_score


def word_cloud_bertopic(topic_model: BERTopic, save_path: Path = None):
    """
    Plots word clouds for BERTopic.

    Parameters
    ----------
    topic_model : BERTopic
        BerTopic model.
    save_path: Path, defaults to None
        Path to save the image file, if None the image is not saved.
    """

    # Get topics
    topics = topic_model.get_topics()
    n_topics = len(topics)

    n_rows = 3
    n_cols = (n_topics + n_rows - 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 9))
    axes = axes.flatten()

    # Loop through the topics and generate a word cloud for each
    for i, topic in enumerate(topics):
        text = {word: value for word, value in topic_model.get_topic(topic)}
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(text)

        # Plot each word cloud in its subplot
        ax = axes[i]
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {i}")

    # Turn off axes for any unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_bertopic_params(topic_model: BERTopic) -> dict:
    """
    Plots word clouds for BERTopic.

    Parameters
    ----------
    topic_model : BERTopic
        BerTopic model.

    Returns
    -------
    dict
        Params for BERTopic model.
    """

    cluster_params = topic_model.hdbscan_model.get_params()
    umap_params = topic_model.umap_model.get_params()
    bertopic_params = topic_model.get_params()

    # UMAP params
    n_components = umap_params["n_components"]
    n_neighbors = umap_params["n_neighbors"]
    min_dist = umap_params["min_dist"]

    # BERTopic params
    nr_topics = bertopic_params["nr_topics"]

    if isinstance(topic_model.hdbscan_model, HDBSCAN):

        # HDBSCAN params
        min_cluster_size = cluster_params["min_cluster_size"]
        max_cluster_size = cluster_params["max_cluster_size"]
        min_samples = cluster_params["min_samples"]
        alpha = cluster_params["alpha"]

        return {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "nr_topics": nr_topics,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
            "min_samples": min_samples,
            "alpha": alpha,
        }

    else:
        # Kmeans params
        n_clusters = cluster_params["n_clusters"]
        n_init = cluster_params["n_init"]
        max_iter = cluster_params["max_iter"]

        return {
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "nr_topics": nr_topics,
            "n_clusters": n_clusters,
            "n_init": n_init,
            "max_iter": max_iter,
        }
