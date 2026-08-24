import pandas as pd


class InsightExplainer:
    """
    Generates explainable business insights
    from analytics outputs.
    """

    def explain_sentiment(
        self,
        sentiment_df
    ):

        if sentiment_df.empty:
            return (
                "No sentiment data available."
            )

        latest = sentiment_df.iloc[-1]

        score = latest["avg_sentiment"]

        if score > 0.3:
            mood = "positive"

        elif score < -0.3:
            mood = "negative"

        else:
            mood = "mixed"

        return (
            f"Customer sentiment is currently "
            f"{mood} with an average sentiment "
            f"score of {score:.2f}."
        )

    def explain_topic_growth(
        self,
        topic_frequency_df
    ):

        if topic_frequency_df.empty:
            return (
                "No topic trend data available."
            )

        latest_month = (
            topic_frequency_df["month"].max()
        )

        latest_df = topic_frequency_df[
            topic_frequency_df["month"]
            == latest_month
        ]

        top_topic = latest_df.sort_values(
            "frequency",
            ascending=False
        ).iloc[0]

        return (
            f"The dominant customer discussion "
            f"topic is '{top_topic['topic_label']}' "
            f"with {top_topic['frequency']} reviews "
            f"in the latest month."
        )

    def explain_drift(
        self,
        similarity_df,
        threshold
    ):

        if similarity_df.empty:
            return (
                "Not enough historical data "
                "for drift analysis."
            )

        latest = similarity_df.iloc[-1]

        similarity = latest[
            "cosine_similarity"
        ]

        if similarity < threshold:

            return (
                "Topic drift detected. "
                "Customer concerns have changed "
                "significantly compared to the "
                "previous month."
            )

        return (
            "Customer discussion patterns "
            "remain relatively stable."
        )