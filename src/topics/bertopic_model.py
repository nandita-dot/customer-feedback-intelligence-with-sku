from bertopic import BERTopic


class BERTopicModel:

    def __init__(self):
        self.model = BERTopic(
            verbose=False
        )

        self.topics = None
        self.probabilities = None
        self.temporal_topics = None

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------

    def fit(self, documents):

        self.topics, self.probabilities = (
            self.model.fit_transform(documents)
        )

    # ---------------------------------------------------
    # GET TOPICS FOR DOCUMENTS
    # ---------------------------------------------------

    def get_topics_for_documents(self, documents):

        if self.topics is not None:
            return self.topics

        topics, _ = self.model.transform(
            documents
        )

        return topics

    # ---------------------------------------------------
    # TEMPORAL TOPIC MODELING
    # ---------------------------------------------------

    def topics_over_time(
        self,
        documents,
        timestamps
    ):

        self.temporal_topics = (
            self.model.topics_over_time(
                documents,
                timestamps,
                evolution_tuning=True,
                global_tuning=True
            )
        )

        return self.temporal_topics

    # ---------------------------------------------------
    # GET KEYWORDS
    # ---------------------------------------------------

    def get_topic_keywords(
        self,
        topic_id,
        top_n=5
    ):

        topic_words = (
            self.model.get_topic(topic_id)
        )

        if topic_words is None:
            return []

        return [
            word
            for word, _ in topic_words[:top_n]
        ]

    # ---------------------------------------------------
    # GET TOPIC INFO
    # ---------------------------------------------------

    def get_topic_info(self):

        return self.model.get_topic_info()