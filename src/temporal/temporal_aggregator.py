import pandas as pd


class TemporalAggregator:

    def aggregate_monthly(
        self,
        df,
        group_by=None
    ):

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["date"]
        )

        df["month"] = (
            df["date"]
            .dt.to_period("M")
            .astype(str)
        )

        # ------------------------------------------------
        # MONTHLY SENTIMENT
        # ------------------------------------------------

        sentiment_group_cols = ["month"]

        if group_by:
            sentiment_group_cols.append(
                group_by
            )

        monthly_sentiment = (
            df.groupby(sentiment_group_cols)
            .agg(
                avg_sentiment=(
                    "compound_score",
                    "mean"
                ),
                negative_reviews=(
                    "sentiment_label",
                    lambda x:
                    (x == "negative").sum()
                ),
                total_reviews=(
                    "review",
                    "count"
                )
            )
            .reset_index()
        )

        # ------------------------------------------------
        # TOPIC FREQUENCY
        # ------------------------------------------------

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
            .reset_index(
                name="frequency"
            )
        )

        # ------------------------------------------------
        # TOPIC SHARE
        # ------------------------------------------------

        monthly_totals = (
            topic_frequencies
            .groupby("month")["frequency"]
            .sum()
            .reset_index(
                name="monthly_total"
            )
        )

        topic_frequencies = (
            topic_frequencies
            .merge(
                monthly_totals,
                on="month",
                how="left"
            )
        )

        topic_frequencies["topic_share"] = (
            topic_frequencies["frequency"]
            / topic_frequencies["monthly_total"]
        )

        # ------------------------------------------------
        # TOPIC SENTIMENT
        # ------------------------------------------------

        topic_sentiment = (
            df.groupby(
                [
                    "month",
                    "topic_id"
                ]
            )
            .agg(
                topic_sentiment=(
                    "compound_score",
                    "mean"
                ),
                negative_ratio=(
                    "sentiment_label",
                    lambda x:
                    (x == "negative").mean()
                )
            )
            .reset_index()
        )

        topic_frequencies = (
            topic_frequencies
            .merge(
                topic_sentiment,
                on=["month", "topic_id"],
                how="left"
            )
        )

        # ------------------------------------------------
        # TOPIC GROWTH
        # ------------------------------------------------

        topic_frequencies = (
            topic_frequencies
            .sort_values(
                ["topic_id", "month"]
            )
        )

        topic_frequencies["previous_frequency"] = (
            topic_frequencies
            .groupby("topic_id")["frequency"]
            .shift(1)
        )

        topic_frequencies["growth_rate"] = (
            (
                topic_frequencies["frequency"]
                -
                topic_frequencies["previous_frequency"]
            )
            /
            topic_frequencies["previous_frequency"]
            .replace(0, 1)
        )

        topic_frequencies["growth_rate"] = (
            topic_frequencies["growth_rate"]
            .replace(
                [float("inf"), -float("inf")],
                0
            )
            .fillna(0)
        )

        # ------------------------------------------------
        # DOMINANT TOPICS
        # ------------------------------------------------

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

        # ------------------------------------------------
        # SKU HOTSPOTS
        # ------------------------------------------------

        sku_topic_frequencies = None

        if "sku" in df.columns:

            sku_topic_frequencies = (
                df.groupby(
                    [
                        "sku",
                        "topic_label"
                    ]
                )
                .agg(
                    count=("review", "count"),
                    avg_sentiment=(
                        "compound_score",
                        "mean"
                    )
                )
                .reset_index()
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