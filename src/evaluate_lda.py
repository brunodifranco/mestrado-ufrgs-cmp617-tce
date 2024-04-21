import pickle
from typing import List, Tuple
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from utils.utils import logger


def get_files(model_path: str) -> Tuple[LdaMulticore, List[str], List, Dictionary]:
    """
    Reads LDA model in pickle format, as well as its corpus, vec and id2word.
    It can be used for plots as well as performance metrics.

    Parameters
    ----------
    model_path: str
        Model path folder name. e.g. "model_5348854098362079"

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
    lda_model: LdaMulticore, vec: List[str], id2word: Dictionary, coherence: str, topn: int
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
    stopwords: List[str], model: LdaMulticore, n_topics: int, max_words: int, figsize: Tuple = (30, 30)
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
    figsize : Tuple, defaults to (30, 30)
        Figure size.
    """

    cloud = WordCloud(
        stopwords=stopwords,
        background_color="white",
        width=2500,
        height=1800,
        max_words=max_words,
        colormap="tab10",
        prefer_horizontal=1.0,
    )

    topics = model.show_topics(formatted=False)
    fig, axes = plt.subplots(1, n_topics, figsize=figsize, sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=400)
        plt.gca().imshow(cloud)
        plt.gca().set_title("Topic " + str(i), fontdict=dict(size=16))
        plt.gca().axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()
