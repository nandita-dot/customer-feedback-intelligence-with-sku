import pandas as pd


class TemporalAggregator:

    def aggregate_monthly(self, df: pd.DataFrame):

        df = df.copy()

        # safety checks
        required = ["date", "compound_score", "topic_id", "sku"]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["month"] = df["date"].dt.to_period("M").astype(str)

        # -----------------------
        # Sentiment trend
        # -----------------------
        monthly_sentiment = (
            df.groupby("month")["compound_score"]
            .mean()
            .reset_index()
            .rename(columns={"compound_score": "avg_sentiment"})
        )

        # -----------------------
        # Topic frequency
        # -----------------------
        topic_frequencies = (
            df.groupby(["month", "topic_id"])
            .size()
            .reset_index(name="frequency")
        )

        # -----------------------
        # SKU + Topic frequency (IMPORTANT)
        # -----------------------
        sku_topic_frequencies = (
            df.groupby(["month", "sku", "topic_id"])
            .size()
            .reset_index(name="frequency")
        )

        # -----------------------
        # Dominant topic per month
        # -----------------------
        dominant_topics = (
            topic_frequencies.loc[
                topic_frequencies.groupby("month")["frequency"].idxmax()
            ]
        )

        return {
            "monthly_sentiment": monthly_sentiment,
            "topic_frequencies": topic_frequencies,
            "sku_topic_frequencies": sku_topic_frequencies,  # 🔥 REQUIRED
            "dominant_topics": dominant_topics
        }