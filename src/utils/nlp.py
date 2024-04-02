import re
from typing import Union, List
from pathlib import Path
import numpy as np
from unidecode import unidecode
from spacy.language import Language
from nltk.stem import RSLPStemmer
from pandas.core.series import Series


def remove_stop_words(
    text: Union[List[str], Series, np.array],
    stop_words_path: Path = "src/utils/stop_words.txt",
) -> List[str]:
    """
    Removes stop words from a list of words.

    Parameters
    ----------
    text: Union[List[str], Series, np.array]
        List, Series, or np.array of words.
    stop_words_path: Path, defaults to "stop_words.txt".
        Path to the file containing stop words.

    Returns
    -------
    List[str]
        List of words with stop words removed.
    """
    stop_words = []

    with open(stop_words_path, "r") as file:
        for row in file:
            stop_words.append(row.strip())

    return [word for word in text if word.lower() not in stop_words]


def preprocess(x: str) -> str:
    """
    Applies NLP preprocess in string.

    Parameters
    ----------
    x: str
        Raw string.

    Returns
    -------
    new_x: str
        String with NLP preprocess applied.
    """

    special_chars = "¨'!#$%&()*+,./:;<=>?@[\]^_`{|}~"
    new_x = x.replace('"', " ")
    for c in special_chars:
        new_x = new_x.replace(c, " ")  # Removes special characters
    new_x = re.sub(r"[^\w\s]", " ", new_x)  # Removes punctuation
    new_x = re.sub("http\S+", " ", new_x)  # Removes links
    new_x = re.sub("@\w+", " ", new_x)  # Removes @
    new_x = re.sub("#\S+", " ", new_x)  # Removes hashtags
    new_x = re.sub("[0-9]+", " ", new_x)  # Removes numbers
    new_x = unidecode(new_x)  # Removes accents
    new_x = re.sub("\s+", " ", new_x)  # Removes spaces

    new_x = " ".join([word for word in new_x.split() if len(word) > 2])

    new_x = new_x.strip()
    return new_x


def stemmer_pt(text: Union[List[str], Series, np.array]) -> List[str]:
    """
    Applies portuguese stemmer in text.

    Parameters
    ----------
    text: Union[List[str], Series, np.array]
        List, Series, or np.array of words.

    Returns
    -------
    List[str]
        List of words with stemmer applied.
    """
    stemmer = RSLPStemmer()
    return [stemmer.stem(word) for word in text]


def lemma_pt(nlp: Language, text: Union[List[str], Series, np.array]) -> List[str]:
    """
    Applies portuguese lemmatization in text.

    Parameters
    ----------
    nlp: Language
        Spacy instance.
    text: Union[List[str], Series, np.array]
        List, Series, or np.array of words.

    Returns
    -------
    List[str]
        List of words with lemmatization applied.
    """

    doc = nlp(" ".join(text))

    return list(set([token.lemma_ for token in doc]))
