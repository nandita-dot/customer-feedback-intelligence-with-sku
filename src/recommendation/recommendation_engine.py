class RecommendationEngine:

    """
    Generates corrective business recommendations
    from detected customer feedback patterns.
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

        "general_feedback":
            "Perform deeper analysis of recurring customer "
            "complaints to identify the underlying issue."
    }

    def generate(
        self,
        severity_df
    ):

        if severity_df.empty:
            return severity_df

        df = severity_df.copy()

        recommendations = []

        for _, row in df.iterrows():

            topic = row["topic_label"]
            severity = row["severity_level"]
            sentiment = row["topic_sentiment"]
            growth = row["growth_rate"]

            recommendation = (
                self.TOPIC_ACTIONS
                .get(
                    topic,
                    self.TOPIC_ACTIONS[
                        "general_feedback"
                    ]
                )
            )

            # --------------------------------------------
            # Severity-based escalation
            # --------------------------------------------

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

            # --------------------------------------------
            # Rapid growth
            # --------------------------------------------

            if growth > 0.50:

                recommendation += (
                    " The issue is experiencing rapid "
                    "growth and should be monitored closely."
                )

            # --------------------------------------------
            # Negative sentiment
            # --------------------------------------------

            if sentiment < -0.30:

                recommendation += (
                    " Strong negative sentiment indicates "
                    "customer dissatisfaction is significant."
                )

            recommendations.append(
                recommendation
            )

        df["recommendation"] = recommendations

        return df