from sklearn.feature_extraction.text import CountVectorizer

vectorizer_model = CountVectorizer(
    # preprocessor=preprocess,
    stop_words=stop_words,
    strip_accents="unicode",
)
# topic_model = BERTopic(vectorizer_model=vectorizer_model)
