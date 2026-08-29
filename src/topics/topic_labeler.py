class TopicLabeler:

    keyword_map = {

        "delivery": [
            "delivery", "shipping",
            "arrive", "tracking",
            "package", "shipment",
            "delay", "delayed"
        ],

        "pricing": [
            "price", "pricing",
            "expensive", "cheap",
            "discount", "cost",
            "money", "value"
        ],

        "quality": [
            "quality", "material",
            "broke", "broken",
            "durable", "defect",
            "damage", "damaged"
        ],

        "support": [
            "support", "service",
            "customer", "respond",
            "staff", "helpful",
            "response"
        ],

        "payment": [
            "payment", "pay",
            "transaction", "card",
            "refund", "checkout",
            "billing"
        ],

        "product_performance": [
            "performance", "slow",
            "fast", "speed",
            "crash", "freeze",
            "lag", "working"
        ],

        "usability": [
            "easy", "difficult",
            "interface", "app",
            "website", "navigation",
            "login"
        ],

        "packaging": [
            "package", "packaging",
            "box", "wrapped",
            "packing"
        ]
    }

    @classmethod
    def generate_label(
        cls,
        topic_words
    ):

        topic_words = [
            str(word).lower()
            for word in topic_words
        ]

        scores = {}

        for label, keywords in (
            cls.keyword_map.items()
        ):

            overlap = len(
                set(topic_words)
                .intersection(
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