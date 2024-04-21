import pickle
from pathlib import Path
from typing import List, Tuple, Union
from torch import Tensor
import numpy as np
from bertopic import BERTopic
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
