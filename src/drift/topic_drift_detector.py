from typing import List

import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity


class TopicDriftDetector:

    """
    Detects temporal concept drift using:

    1. Topic distribution change
    2. Sentiment change
    3. Review volume change

    Cosine similarity remains the primary
    semantic distribution similarity measure.
    """

    def __init__(
        self,
        threshold=0.80
    ):

        self.threshold = threshold

    # ---------------------------------------------------
    # BUILD TOPIC DISTRIBUTION
    # ---------------------------------------------------

    def build_topic_distribution(
        self,
        df
    ):

        distribution = (
            df.groupby(
                ["month", "topic_id"]
            )
            .size()
            .unstack(
                fill_value=0
            )
        )

        distribution = distribution.div(
            distribution.sum(axis=1),
            axis=0
        )

        return distribution

    # ---------------------------------------------------
    # COSINE SIMILARITY
    # ---------------------------------------------------

    def compute_similarity(
        self,
        topic_distribution
    ):

        months = (
            topic_distribution
            .index
            .tolist()
        )

        results = []

        for i in range(1, len(months)):

            previous_month = months[i - 1]
            current_month = months[i]

            prev_vector = (
                topic_distribution
                .loc[previous_month]
                .values
                .reshape(1, -1)
            )

            curr_vector = (
                topic_distribution
                .loc[current_month]
                .values
                .reshape(1, -1)
            )

            similarity = cosine_similarity(
                prev_vector,
                curr_vector
            )[0][0]

            topic_drift = (
                1 - similarity
            )

            results.append({

                "previous_month":
                    previous_month,

                "current_month":
                    current_month,

                "cosine_similarity":
                    round(similarity, 4),

                "topic_drift":
                    round(topic_drift, 4),

                "drift_detected":
                    similarity < self.threshold
            })

        return pd.DataFrame(results)

    # ---------------------------------------------------
    # SENTIMENT DRIFT
    # ---------------------------------------------------

    def compute_sentiment_drift(
        self,
        df
    ):

        monthly = (
            df.groupby("month")
            .agg(
                avg_sentiment=(
                    "compound_score",
                    "mean"
                ),
                review_count=(
                    "review",
                    "count"
                )
            )
            .reset_index()
        )

        monthly["sentiment_change"] = (
            monthly["avg_sentiment"]
            .diff()
            .abs()
        )

        monthly["volume_change"] = (
            monthly["review_count"]
            .pct_change()
            .abs()
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        return monthly

    # ---------------------------------------------------
    # COMBINED DRIFT
    # ---------------------------------------------------

    def build_combined_drift(
        self,
        similarity_df,
        sentiment_df
    ):

        if similarity_df.empty:
            return similarity_df

        result = similarity_df.copy()

        sentiment_lookup = (
            sentiment_df[
                [
                    "month",
                    "sentiment_change",
                    "volume_change"
                ]
            ]
            .rename(
                columns={
                    "month":
                        "current_month"
                }
            )
        )

        result = result.merge(
            sentiment_lookup,
            on="current_month",
            how="left"
        )

        result["sentiment_change"] = (
            result["sentiment_change"]
            .fillna(0)
        )

        result["volume_change"] = (
            result["volume_change"]
            .fillna(0)
        )

        # Normalize sentiment and volume changes
        max_sentiment = max(
            result["sentiment_change"].max(),
            1e-9
        )

        max_volume = max(
            result["volume_change"].max(),
            1e-9
        )

        result["sentiment_drift"] = (
            result["sentiment_change"]
            / max_sentiment
        )

        result["volume_drift"] = (
            result["volume_change"]
            / max_volume
        )

        # Combined Concept Drift Score
        result["concept_drift_score"] = (
            0.60 *
            result["topic_drift"]
            +
            0.25 *
            result["sentiment_drift"]
            +
            0.15 *
            result["volume_drift"]
        )

        result["concept_drift_detected"] = (
            result["concept_drift_score"] >= 0.35
        )

        return result

    # ---------------------------------------------------
    # ALERT GENERATION
    # ---------------------------------------------------

    def generate_alerts(
        self,
        drift_df
    ):

        alerts = []

        if drift_df.empty:
            return alerts

        for _, row in drift_df.iterrows():

            if row["concept_drift_detected"]:

                alerts.append(
                    f"[ALERT] Significant customer "
                    f"feedback drift detected between "
                    f"{row['previous_month']} and "
                    f"{row['current_month']} "
                    f"(drift score="
                    f"{row['concept_drift_score']:.2f})"
                )

        return alerts

    # ---------------------------------------------------
    # COMPLETE PIPELINE
    # ---------------------------------------------------

    def detect_drift(
        self,
        df
    ):

        topic_distribution = (
            self.build_topic_distribution(df)
        )

        similarity_df = (
            self.compute_similarity(
                topic_distribution
            )
        )

        sentiment_df = (
            self.compute_sentiment_drift(df)
        )

        combined_df = (
            self.build_combined_drift(
                similarity_df,
                sentiment_df
            )
        )

        alerts = (
            self.generate_alerts(
                combined_df
            )
        )

        return {

            "topic_distribution":
                topic_distribution,

            "similarity_scores":
                combined_df,

            "alerts":
                alerts
        }