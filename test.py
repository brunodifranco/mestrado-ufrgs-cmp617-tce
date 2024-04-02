topic_model = BERTopic(
    nr_topics=5,
    zeroshot_topic_list=[
        "vehicles",
        "construction",
        "health/hospital",
        "education",
        "food",
    ],
    min_topic_size=5,
    language="brazilian portuguese",
    low_memory=True,
)
topics, _ = topic_model.fit_transform(docs)
