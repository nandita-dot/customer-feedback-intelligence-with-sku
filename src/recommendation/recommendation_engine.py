class RecommendationEngine:
    """
    Generates context-aware corrective business recommendations
    from detected customer feedback patterns.

    The engine combines:
        - business topic
        - severity
        - sentiment
        - growth
        - concept drift

    The recommendation remains deterministic and explainable.
    """

    # =========================================================
    # BASE BUSINESS ACTIONS
    # =========================================================

    TOPIC_ACTIONS = {

        "delivery":
            "Investigate logistics and delivery partners. "
            "Review delayed shipments and improve tracking visibility.",

        "pricing":
            "Review pricing competitiveness, discounts and "
            "customer value perception.",

        "quality":
            "Investigate product quality and manufacturing issues. "
            "Perform quality-control checks on affected products.",

        "support":
            "Review customer support response times and "
            "increase support capacity for recurring complaints.",

        "payment":
            "Review payment processing failures, transaction "
            "issues and billing handling.",

        "product_performance":
            "Investigate product performance problems and "
            "identify recurring technical failures.",

        "usability":
            "Review product usability, interface and "
            "customer interaction difficulties.",

        "packaging":
            "Review packaging quality, protection and "
            "packing procedures.",

        "general_feedback":
            "Perform deeper analysis of recurring customer "
            "complaints to identify the underlying issue."
    }

    # =========================================================
    # CONDITION-BASED ACTIONS
    # =========================================================

    SEVERITY_ACTIONS = {

        "Critical":
            "Escalate the issue immediately to the responsible "
            "operations or product team.",

        "High":
            "Prioritize investigation within the current "
            "review cycle.",

        "Medium":
            "Monitor the issue and include it in the next "
            "improvement cycle."
    }

    GROWTH_ACTION = (
        "Complaint volume is increasing rapidly, so monitor "
        "the issue closely and investigate the underlying cause."
    )

    NEGATIVE_SENTIMENT_ACTION = (
        "Strong negative sentiment indicates significant "
        "customer dissatisfaction and requires focused attention."
    )

    DRIFT_ACTION = (
        "The issue's characteristics are changing over time, "
        "so review recent complaints for emerging failure patterns."
    )

    # =========================================================
    # GENERATE RECOMMENDATIONS
    # =========================================================

    def generate(self, severity_df):

        if severity_df.empty:
            return severity_df

        df = severity_df.copy()

        # =====================================================
        # NORMALIZE TOPIC LABELS
        # =====================================================

        df["topic_label"] = (
            df["topic_label"]
            .fillna("general_feedback")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # =====================================================
        # REMOVE DUPLICATE BUSINESS ISSUES
        #
        # Multiple BERTopic IDs may map to the same
        # business-friendly label.
        #
        # Keep the most severe occurrence.
        # =====================================================

        df = (
            df
            .sort_values(
                "severity_score",
                ascending=False
            )
            .drop_duplicates(
                subset=["month", "topic_label"],
                keep="first"
            )
            .reset_index(drop=True)
        )

        # =====================================================
        # GENERATE CONTEXT-AWARE RECOMMENDATIONS
        # =====================================================

        recommendations = []

        for _, row in df.iterrows():

            topic = row["topic_label"]

            severity = row.get(
                "severity_level",
                "Medium"
            )

            sentiment = row.get(
                "topic_sentiment",
                0
            )

            growth = row.get(
                "growth_rate",
                0
            )

            drift = row.get(
                "concept_drift_score",
                0
            )

            # -------------------------------------------------
            # Normalize numeric values
            # -------------------------------------------------

            try:
                sentiment = float(sentiment)
            except (TypeError, ValueError):
                sentiment = 0.0

            try:
                growth = float(growth)
            except (TypeError, ValueError):
                growth = 0.0

            try:
                drift = float(drift)
            except (TypeError, ValueError):
                drift = 0.0

            # -------------------------------------------------
            # BASE TOPIC ACTION
            # -------------------------------------------------

            recommendation = self.TOPIC_ACTIONS.get(
                topic,
                self.TOPIC_ACTIONS["general_feedback"]
            )

            # -------------------------------------------------
            # SEVERITY
            # -------------------------------------------------

            severity_action = self.SEVERITY_ACTIONS.get(
                severity
            )

            if severity_action:
                recommendation += " " + severity_action

            # -------------------------------------------------
            # RAPID GROWTH
            # -------------------------------------------------

            if growth > 0.50:
                recommendation += " " + self.GROWTH_ACTION

            # -------------------------------------------------
            # NEGATIVE SENTIMENT
            # -------------------------------------------------

            if sentiment < -0.30:
                recommendation += " " + self.NEGATIVE_SENTIMENT_ACTION

            # -------------------------------------------------
            # CONCEPT DRIFT
            # -------------------------------------------------

            if drift >= 0.80:
                recommendation += " " + self.DRIFT_ACTION

            recommendations.append(recommendation)

        # =====================================================
        # ADD RECOMMENDATION COLUMN
        # =====================================================

        df["recommendation"] = recommendations

        # =====================================================
        # FINAL PRIORITY ORDER
        # =====================================================

        df = (
            df
            .sort_values(
                "severity_score",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return df