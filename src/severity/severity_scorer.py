import pandas as pd
import numpy as np


class SeverityScorer:

    """
    Calculates a Severity Impact Score (SIS)
    for customer issues.

    Factors:

    - Negative sentiment
    - Topic volume
    - Topic growth
    - Drift
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
    # CALCULATE SEVERITY
    # ---------------------------------------------------

    def calculate(
        self,
        topic_df,
        drift_df
    ):

        if topic_df.empty:
            return pd.DataFrame()

        df = topic_df.copy()

        # ------------------------------------------------
        # NEGATIVITY
        # ------------------------------------------------

        df["negative_impact"] = (
            df["negative_ratio"]
            .clip(0, 1)
        )

        # ------------------------------------------------
        # VOLUME
        # ------------------------------------------------

        df["volume_score"] = (
            self.normalize(
                df["frequency"]
            )
        )

        # ------------------------------------------------
        # GROWTH
        # ------------------------------------------------

        df["growth_score"] = (
            self.normalize(
                df["growth_rate"]
                .clip(lower=0)
            )
        )

        # ------------------------------------------------
        # DRIFT
        # ------------------------------------------------

        drift_lookup = (
            drift_df[
                [
                    "current_month",
                    "concept_drift_score"
                ]
            ]
            .rename(
                columns={
                    "current_month":
                        "month"
                }
            )
        )

        df = df.merge(
            drift_lookup,
            on="month",
            how="left"
        )

        df["concept_drift_score"] = (
            df["concept_drift_score"]
            .fillna(0)
        )

        # ------------------------------------------------
        # SEVERITY IMPACT SCORE
        # ------------------------------------------------

        df["severity_score"] = (

            self.sentiment_weight
            * df["negative_impact"]

            +

            self.volume_weight
            * df["volume_score"]

            +

            self.growth_weight
            * df["growth_score"]

            +

            self.drift_weight
            * df["concept_drift_score"]
        )

        # ------------------------------------------------
        # SCALE TO 0-100
        # ------------------------------------------------

        df["severity_score"] = (
            df["severity_score"] * 100
        ).round(2)

        # ------------------------------------------------
        # SEVERITY CATEGORY
        # ------------------------------------------------

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

        return df.sort_values(
            "severity_score",
            ascending=False
        )
