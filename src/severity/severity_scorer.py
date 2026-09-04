import pandas as pd
import numpy as np


class SeverityScorer:
    """
    Calculates a Severity Impact Score (SIS)
    for customer business issues.

    Multiple BERTopic topic IDs may have the same
    business-friendly topic label.

    Example:

        Topic 2 -> delivery
        Topic 5 -> delivery

    These are combined into one business issue
    before calculating severity.
    """

    def __init__(
        self,
        sentiment_weight=0.35,
        volume_weight=0.20,
        growth_weight=0.20,
        drift_weight=0.25
    ):

        self.sentiment_weight = sentiment_weight
        self.volume_weight = volume_weight
        self.growth_weight = growth_weight
        self.drift_weight = drift_weight

    # ---------------------------------------------------
    # MIN-MAX NORMALIZATION
    # ---------------------------------------------------

    @staticmethod
    def normalize(series):

        minimum = series.min()
        maximum = series.max()

        if maximum == minimum:

            return pd.Series(
                0.5,
                index=series.index
            )

        return (
            (series - minimum)
            /
            (maximum - minimum)
        )

    # ---------------------------------------------------
    # COMBINE SAME BUSINESS ISSUES
    # ---------------------------------------------------

    @staticmethod
    def aggregate_business_issues(topic_df):

        if topic_df.empty:
            return pd.DataFrame()

        df = topic_df.copy()

        # Make sure labels are consistent
        df["topic_label"] = (
            df["topic_label"]
            .fillna("general_feedback")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # ------------------------------------------------
        # Combine multiple BERTopic IDs having the same
        # business label.
        # ------------------------------------------------

        grouped = (
            df.groupby(
                ["month", "topic_label"],
                as_index=False
            )
            .agg(
                frequency=(
                    "frequency",
                    "sum"
                ),

                negative_reviews=(
                    "negative_ratio",
                    lambda x: 0
                ),

                topic_sentiment=(
                    "topic_sentiment",
                    "mean"
                ),

                negative_ratio=(
                    "negative_ratio",
                    "mean"
                )
            )
        )

        # ------------------------------------------------
        # Recalculate business-level growth.
        # ------------------------------------------------

        grouped = (
            grouped
            .sort_values(
                ["topic_label", "month"]
            )
            .reset_index(drop=True)
        )

        grouped["previous_frequency"] = (
            grouped
            .groupby("topic_label")["frequency"]
            .shift(1)
        )

        grouped["growth_rate"] = (
            (
                grouped["frequency"]
                -
                grouped["previous_frequency"]
            )
            /
            grouped["previous_frequency"]
            .replace(0, 1)
        )

        grouped["growth_rate"] = (
            grouped["growth_rate"]
            .replace(
                [np.inf, -np.inf],
                0
            )
            .fillna(0)
        )

        return grouped

    # ---------------------------------------------------
    # CALCULATE SEVERITY
    # ---------------------------------------------------

    def calculate(
        self,
        topic_df,
        drift_df
    ):

        if topic_df.empty:
            return pd.DataFrame()

        # =================================================
        # BUSINESS-LEVEL AGGREGATION
        # =================================================

        df = self.aggregate_business_issues(
            topic_df
        )

        if df.empty:
            return pd.DataFrame()

        # =================================================
        # NEGATIVE SENTIMENT
        # =================================================

        df["negative_impact"] = (
            df["negative_ratio"]
            .clip(0, 1)
        )

        # =================================================
        # VOLUME
        # =================================================

        df["volume_score"] = (
            self.normalize(
                df["frequency"]
            )
        )

        # =================================================
        # GROWTH
        # =================================================

        df["growth_score"] = (
            self.normalize(
                df["growth_rate"]
                .clip(lower=0)
            )
        )

        # =================================================
        # DRIFT
        # =================================================

        if (
            drift_df is not None
            and not drift_df.empty
            and "concept_drift_score" in drift_df.columns
        ):

            drift_lookup = (
                drift_df[
                    [
                        "current_month",
                        "concept_drift_score"
                    ]
                ]
                .rename(
                    columns={
                        "current_month": "month"
                    }
                )
            )

            # There should normally be one drift score
            # per month.
            drift_lookup = (
                drift_lookup
                .drop_duplicates(
                    subset=["month"]
                )
            )

            df = df.merge(
                drift_lookup,
                on="month",
                how="left"
            )

        else:

            df["concept_drift_score"] = 0

        df["concept_drift_score"] = (
            df["concept_drift_score"]
            .fillna(0)
            .clip(0, 1)
        )

        # =================================================
        # SEVERITY IMPACT SCORE
        # =================================================

        df["severity_score"] = (

            self.sentiment_weight
            *
            df["negative_impact"]

            +

            self.volume_weight
            *
            df["volume_score"]

            +

            self.growth_weight
            *
            df["growth_score"]

            +

            self.drift_weight
            *
            df["concept_drift_score"]
        )

        # =================================================
        # SCALE 0-100
        # =================================================

        df["severity_score"] = (
            df["severity_score"]
            * 100
        ).round(2)

        # =================================================
        # SEVERITY CATEGORY
        # =================================================

        def classify(score):

            if score >= 75:
                return "Critical"

            if score >= 50:
                return "High"

            if score >= 25:
                return "Medium"

            return "Low"

        df["severity_level"] = (
            df["severity_score"]
            .apply(classify)
        )

        # =================================================
        # FINAL SORT
        # =================================================

        return (
            df
            .sort_values(
                [
                    "month",
                    "severity_score"
                ],
                ascending=[
                    True,
                    False
                ]
            )
            .reset_index(drop=True)
        )