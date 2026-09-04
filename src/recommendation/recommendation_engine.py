class RecommendationEngine:
    """
    Generates corrective business recommendations
    from detected customer feedback patterns.

    Important:
    Multiple BERTopic topic IDs can receive the same
    business-friendly label.

    Example:

        Topic 3  -> delivery
        Topic 7  -> delivery

    For recommendations, these must be treated as
    ONE business issue rather than separate issues.
    """

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
            "issues and refund handling.",

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
        # IMPORTANT FIX
        #
        # Several BERTopic IDs may have the same business
        # label.
        #
        # For example:
        #
        # topic_id 3  -> delivery
        # topic_id 8  -> delivery
        # topic_id 12 -> delivery
        #
        # Recommendations must contain only ONE delivery.
        #
        # We keep the most severe occurrence because the
        # recommendation section is intended to prioritize
        # the most important business issues.
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
        # GENERATE RECOMMENDATIONS
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

            # -------------------------------------------------
            # Base recommendation
            # -------------------------------------------------

            recommendation = self.TOPIC_ACTIONS.get(
                topic,
                self.TOPIC_ACTIONS["general_feedback"]
            )

            # -------------------------------------------------
            # Severity-based escalation
            # -------------------------------------------------

            if severity == "Critical":

                recommendation = (
                    "URGENT: "
                    + recommendation
                    + " Escalate the issue to "
                    "the responsible operations/product team."
                )

            elif severity == "High":

                recommendation = (
                    recommendation
                    + " Prioritize investigation "
                    "within the current review cycle."
                )

            # -------------------------------------------------
            # Rapid growth
            # -------------------------------------------------

            try:
                growth = float(growth)
            except (TypeError, ValueError):
                growth = 0.0

            if growth > 0.50:

                recommendation += (
                    " The issue is experiencing rapid "
                    "growth and should be monitored closely."
                )

            # -------------------------------------------------
            # Negative sentiment
            # -------------------------------------------------

            try:
                sentiment = float(sentiment)
            except (TypeError, ValueError):
                sentiment = 0.0

            if sentiment < -0.30:

                recommendation += (
                    " Strong negative sentiment indicates "
                    "customer dissatisfaction is significant."
                )

            recommendations.append(
                recommendation
            )

        # =====================================================
        # ADD RECOMMENDATION COLUMN
        # =====================================================

        df["recommendation"] = recommendations

        # =====================================================
        # FINAL SORT
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