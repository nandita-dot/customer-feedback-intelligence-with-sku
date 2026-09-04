class TopicLabeler:

    # =========================================================
    # KEYWORD DEFINITIONS
    # =========================================================

    keyword_map = {

        "delivery": [
            "delivery",
            "deliver",
            "delivered",
            "shipping",
            "ship",
            "shipment",
            "courier",
            "tracking",
            "track",
            "arrival",
            "arrive",
            "arrived",
            "delay",
            "delayed",
            "late",
            "lateness",
            "dispatch",
            "dispatched"
        ],

        "pricing": [
            "price",
            "pricing",
            "expensive",
            "cheap",
            "discount",
            "cost",
            "money",
            "value",
            "worth",
            "affordable",
            "fee",
            "charge"
        ],

        "quality": [
            "quality",
            "material",
            "broke",
            "broken",
            "durable",
            "durability",
            "defect",
            "defective",
            "damage",
            "damaged",
            "poor",
            "faulty",
            "failure"
        ],

        "support": [
            "support",
            "customer_support",
            "respond",
            "responded",
            "response",
            "staff",
            "helpful",
            "assistance",
            "help",
            "agent",
            "representative"
        ],

        "payment": [
            "payment",
            "pay",
            "paid",
            "transaction",
            "card",
            "refund",
            "checkout",
            "billing",
            "bill",
            "charge",
            "charged",
            "invoice"
        ],

        "product_performance": [
            "performance",
            "slow",
            "fast",
            "speed",
            "crash",
            "crashed",
            "freeze",
            "frozen",
            "lag",
            "laggy",
            "working",
            "works",
            "function",
            "functioning",
            "battery",
            "overheat"
        ],

        "usability": [
            "easy",
            "difficult",
            "interface",
            "app",
            "application",
            "website",
            "navigation",
            "login",
            "password",
            "account",
            "screen",
            "design",
            "usable",
            "usability"
        ],

        "packaging": [
            "packaging",
            "box",
            "wrapped",
            "wrapping",
            "packing",
            "packed",
            "package"
        ]
    }

    # =========================================================
    # STRONG KEYWORDS
    # These words are much more specific to a topic.
    # =========================================================

    strong_keywords = {

        "delivery": [
            "delivery",
            "delivered",
            "shipping",
            "shipment",
            "courier",
            "tracking",
            "delay",
            "delayed",
            "late",
            "arrival",
            "arrived",
            "dispatch"
        ],

        "pricing": [
            "price",
            "pricing",
            "expensive",
            "discount",
            "cost",
            "value",
            "affordable"
        ],

        "quality": [
            "quality",
            "defect",
            "defective",
            "broken",
            "broke",
            "damaged",
            "damage",
            "durable",
            "faulty"
        ],

        "support": [
            "support",
            "response",
            "respond",
            "staff",
            "agent",
            "assistance",
            "helpful"
        ],

        "payment": [
            "payment",
            "transaction",
            "refund",
            "checkout",
            "billing",
            "invoice",
            "charged"
        ],

        "product_performance": [
            "performance",
            "crash",
            "freeze",
            "lag",
            "battery",
            "overheat",
            "speed"
        ],

        "usability": [
            "interface",
            "website",
            "navigation",
            "login",
            "password",
            "account",
            "usability"
        ],

        "packaging": [
            "packaging",
            "box",
            "wrapped",
            "wrapping",
            "packing",
            "packed"
        ]
    }

    # =========================================================
    # AMBIGUOUS WORDS
    # These should not strongly influence the label.
    # =========================================================

    ambiguous_keywords = {
        "package",
        "customer",
        "service",
        "working",
        "works",
        "fast",
        "slow",
        "help",
        "charge"
    }

    # =========================================================
    # LABEL GENERATION
    # =========================================================

    @classmethod
    def generate_label(cls, topic_words):

        # -----------------------------------------------------
        # Handle empty / invalid input
        # -----------------------------------------------------

        if topic_words is None:
            return "general_feedback"

        if isinstance(topic_words, str):
            topic_words = topic_words.split()

        if not topic_words:
            return "general_feedback"

        # -----------------------------------------------------
        # Normalize words
        # -----------------------------------------------------

        normalized_words = []

        for word in topic_words:

            word = str(word).strip().lower()

            # Remove simple punctuation
            word = (
                word
                .replace(",", "")
                .replace(".", "")
                .replace("!", "")
                .replace("?", "")
                .replace(":", "")
                .replace(";", "")
                .replace("-", "_")
            )

            if word:
                normalized_words.append(word)

        # Remove duplicates while preserving order
        topic_words = list(dict.fromkeys(normalized_words))

        # -----------------------------------------------------
        # Calculate scores
        # -----------------------------------------------------

        scores = {
            label: 0
            for label in cls.keyword_map
        }

        for label, keywords in cls.keyword_map.items():

            for word in topic_words:

                # Strong keyword
                if word in cls.strong_keywords.get(label, []):

                    scores[label] += 3

                # Normal keyword
                elif word in keywords:

                    # Ambiguous words get very low weight
                    if word in cls.ambiguous_keywords:
                        scores[label] += 0.5
                    else:
                        scores[label] += 1

        # -----------------------------------------------------
        # No matching topic
        # -----------------------------------------------------

        max_score = max(scores.values())

        if max_score == 0:
            return "general_feedback"

        # -----------------------------------------------------
        # Find candidates with highest score
        # -----------------------------------------------------

        candidates = [
            label
            for label, score in scores.items()
            if score == max_score
        ]

        # -----------------------------------------------------
        # Tie-breaking
        #
        # This is particularly important for:
        #
        # package → packaging
        # package + delivery → delivery
        #
        # A generic "package" should NOT automatically
        # become delivery.
        # -----------------------------------------------------

        if len(candidates) > 1:

            # Count strong matches for each candidate
            strong_counts = {}

            for label in candidates:

                strong_counts[label] = sum(
                    1
                    for word in topic_words
                    if word in cls.strong_keywords.get(
                        label,
                        []
                    )
                )

            max_strong = max(
                strong_counts.values()
            )

            strong_candidates = [
                label
                for label in candidates
                if strong_counts[label] == max_strong
            ]

            if len(strong_candidates) == 1:
                return strong_candidates[0]

            candidates = strong_candidates

        # -----------------------------------------------------
        # Explicit priority for remaining ties
        # -----------------------------------------------------

        priority = [
            "delivery",
            "pricing",
            "quality",
            "support",
            "payment",
            "product_performance",
            "usability",
            "packaging"
        ]

        for label in priority:

            if label in candidates:
                return label

        # -----------------------------------------------------
        # Final fallback
        # -----------------------------------------------------

        return "general_feedback"