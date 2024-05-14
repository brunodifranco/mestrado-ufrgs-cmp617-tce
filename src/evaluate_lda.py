import os
import json
import pickle
from typing import List, Tuple, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from utils.utils import logger, pct_format


def read_json_metrics(base_path: Path, model_type: str) -> List[Dict]:
    """
    Reads json metrics in a List[Dict].

    Parameters
    ----------
    base_path: Path
        LDA (Latent Dirichlet Allocation) fit model.
    model_type: str
        Model type. Either "lemmatization" or "stemmer".

    Returns
    --------
    metrics_list : List[Dict]
        Metrics list.
    """

    metrics_list = []
    for top_n in range(5, 11):
        file_path = os.path.join(
            base_path,
            model_type,
            f"best_model_topn_{top_n}",
            "metrics",
            "lda_metrics.json",
        )
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
                data["mean_metrics"]["top_n"] = top_n
                metrics_list.append(data["mean_metrics"])

    return metrics_list


def get_files(model_path: str) -> Tuple[LdaMulticore, List[str], List, Dictionary]:
    """
    Reads LDA model in pickle format, as well as its corpus, vec and id2word.
    It can be used for plots as well as performance metrics.

    Parameters
    ----------
    model_path: str
        Model path folder name.

    Returns
    --------
    lda_model: LdaMulticore
        LDA (Latent Dirichlet Allocation) fit model.
    vec : List[str]
        List of text data for model optimization.
    corpus : List
        Corpus.
    id2word : Dictionary
        Gensim Dictionary.
    """

    lda_model = pickle.load(open(model_path + "/model.pkl", "rb"))
    vec = pickle.load(open(model_path + "/vec.pkl", "rb"))
    corpus = pickle.load(open(model_path + "/corpus.pkl", "rb"))
    id2word = pickle.load(open(model_path + "/id2word.pkl", "rb"))

    logger.info(f"Model and its files loaded")

    return lda_model, vec, corpus, id2word


def coherence_score(
    lda_model: LdaMulticore,
    vec: List[str],
    id2word: Dictionary,
    coherence: str,
    topn: int,
) -> float:
    """
    Calculates the desired coherence_score.

    Parameters
    ----------
    lda_model: LdaMulticore
        LDA (Latent Dirichlet Allocation) saved model.
    vec : List[str]
        List of text data for model optimization.
    id2word : Dictionary
        Gensim Dictionary.
    coherence : str
        Desired coherence metrics. Options are {"c_v", "c_npmi"}
    topn : int
        Number of top words to be extracted from each topic.

    Returns
    --------
    coherence_lda: float
        Coherence Score.
    """

    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=vec, dictionary=id2word, coherence=coherence, topn=topn
    )
    coherence_lda = coherence_model_lda.get_coherence()

    logger.info(f"Coherence score {coherence} calculated")

    return coherence_lda


def word_cloud(
    stopwords: List[str],
    model: LdaMulticore,
    n_topics: int,
    max_words: int,
    figsize: Tuple[int, int] = (30, 15),
    save_path: Path = None,
):
    """
    Displays word cloud.

    Parameters
    ----------
    stopwords: List[str]
        List with stopwords.
    lda_model: LdaMulticore
        LDA (Latent Dirichlet Allocation) saved model.
    n_topics: int
        Number of topics
    max_words: int
        Top n words that represent the topic
    figsize : Tuple[int, int], defaults to (30, 30)
        Figure size.
    save_path: Path, defaults to None
        Path to save the image file, if None the image is not saved.
    """

    cloud = WordCloud(
        stopwords=stopwords,
        background_color="white",
        width=2500,
        height=1800,
        max_words=max_words,
        prefer_horizontal=1.0,
    )

    # Calculate the number of columns and rows needed
    n_cols = n_topics // 2 + n_topics % 2  # Ensure even distribution
    n_rows = 2  # Set the number of rows

    topics = model.show_topics(formatted=False)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)

    axes = axes.flatten()
    for i, ax in enumerate(axes[:n_topics]):
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=400)
        ax.imshow(cloud)
        ax.set_title("Topic " + str(i), fontdict=dict(size=40))
        ax.axis("off")

    # Turn off any unused axes
    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()


def get_topic_dist(lda_model: LdaMulticore, corpus: List) -> List[int]:
    """
    Generates topic distribution for lda_model.

    Parameters
    ----------
    lda_model: LdaMulticore
        LDA (Latent Dirichlet Allocation) saved model.
    corpus : List
        Corpus.

    Returns
    --------
    topic_dist: List[int]
        Topic distribution.
    """
    get_document_topics = [lda_model.get_document_topics(item) for item in corpus]
    topic_dist = []

    for doc_topics in get_document_topics:
        dominant_topic = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0][0]
        topic_dist.append(dominant_topic)

    return topic_dist


def plot_lda_topic_dist(
    topic_dist: List[int],
    save_path: Path = None,
    fontsize: int = 8,
    figsize: Tuple[int, int] = (18, 10),
):
    """
    Plots pie chart for LDA topic model distribution.

    Parameters
    ----------
    topic_dist: List[int]
        Topic distribution list.
    save_path: Path, defaults to None
        Path to save the image file, if None the image is not saved.
    fontsize : int, defaults to 8
        Font size.
    figsiz: Tuple[int, int], defaults to (18, 10)
        Figure size.
    """

    # Getting the frequency for each topic
    topic_counts = Counter(topic_dist)
    labels = [f"Topic {i}" for i in topic_counts.keys()]
    sizes = list(topic_counts.values())

    explode = [0.05] * len(labels)
    wp = {"linewidth": 0.7, "edgecolor": "black"}

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    _, _, autotexts = ax.pie(
        sizes,
        autopct=lambda pct: pct_format(pct, sizes),
        explode=explode,
        labels=labels,
        shadow=True,
        startangle=90,
        wedgeprops=wp,
        textprops=dict(color="black"),
    )

    plt.setp(autotexts, fontsize=fontsize, weight="bold")

    if save_path:
        plt.savefig(save_path)
