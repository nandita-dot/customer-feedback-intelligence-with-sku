# src/temporal/temporal_aggregator.py

import pandas as pd


class TemporalAggregator:

    def aggregate_monthly(
        self,
        df: pd.DataFrame,
        group_by: str = None
    ):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(subset=["date"])

        df["month"] = (
            df["date"]
            .dt.to_period("M")
            .astype(str)
        )

        # ----------------------------------------
        # Monthly sentiment
        # ----------------------------------------

        sentiment_group_cols = ["month"]

        if group_by:
            sentiment_group_cols.append(
                group_by
            )

        monthly_sentiment = (
            df.groupby(sentiment_group_cols)[
                "compound_score"
            ]
            .mean()
            .reset_index()
        )

        monthly_sentiment.rename(
            columns={
                "compound_score":
                "avg_sentiment"
            },
            inplace=True
        )

        # ----------------------------------------
        # Topic frequencies
        # ----------------------------------------

        topic_group_cols = [
            "month",
            "topic_id",
            "topic_label"
        ]

        if group_by:
            topic_group_cols.insert(
                1,
                group_by
            )

        topic_frequencies = (
            df.groupby(topic_group_cols)
            .size()
            .reset_index(name="frequency")
        )

        # ----------------------------------------
        # Dominant topics
        # ----------------------------------------

        dominant_topics = (
            topic_frequencies
            .sort_values(
                "frequency",
                ascending=False
            )
            .groupby("month")
            .first()
            .reset_index()
        )

        # ----------------------------------------
        # SKU Hotspots
        # ----------------------------------------

        sku_topic_frequencies = None

        if "sku" in df.columns:

            sku_topic_frequencies = (
                df.groupby(
                    [
                        "sku",
                        "topic_label"
                    ]
                )
                .size()
                .reset_index(name="count")
                .sort_values(
                    "count",
                    ascending=False
                )
            )

        return {
            "monthly_sentiment":
            monthly_sentiment,

            "topic_frequencies":
            topic_frequencies,

            "dominant_topics":
            dominant_topics,

            "sku_topic_frequencies":
            sku_topic_frequencies
        }