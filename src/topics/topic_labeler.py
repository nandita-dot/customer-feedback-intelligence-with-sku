class TopicLabeler:
    """
    Converts raw topic keywords into
    business-friendly labels.
    """

    keyword_map = {
        "delivery": [
            "delivery",
            "shipping",
            "arrive",
            "tracking",
            "package"
        ],

        "pricing": [
            "price",
            "pricing",
            "affordable",
            "expensive",
            "discount",
            "money",
            "value"
        ],

        "quality": [
            "quality",
            "premium",
            "material",
            "broke",
            "cheap",
            "durable"
        ],

        "support": [
            "support",
            "service",
            "customer",
            "respond",
            "staff",
            "helpful"
        ]
    }

    @classmethod
    def generate_label(cls, topic_words):
        """
        Generate label from topic keywords.
        """

        topic_words = [
            str(word).lower()
            for word in topic_words
        ]

        scores = {}

        for label, keywords in cls.keyword_map.items():

            overlap = len(
                set(topic_words).intersection(
                    set(keywords)
                )
            )

            scores[label] = overlap

        best_label = max(
            scores,
            key=scores.get
        )

        if scores[best_label] == 0:
            return "general_feedback"

        return best_label