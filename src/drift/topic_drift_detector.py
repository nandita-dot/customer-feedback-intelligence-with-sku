from typing import List
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

print("LOADING TOPIC DRIFT DETECTOR")

class TopicDriftDetector:
    """
    Detects topic distribution drift across months
    using cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.75
    ):

        self.threshold = threshold

    def build_topic_distribution(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build normalized topic distributions
        for each month.
        """

        # -----------------------------------
        # Count topic frequencies
        # -----------------------------------

        distribution = (
            df.groupby(["month", "topic_id"])
            .size()
            .unstack(fill_value=0)
        )

        # -----------------------------------
        # Normalize rows
        # -----------------------------------

        distribution = distribution.div(
            distribution.sum(axis=1),
            axis=0
        )

        return distribution

    def compute_similarity(
        self,
        topic_distribution: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute cosine similarity between
        consecutive months.
        """

        months = topic_distribution.index.tolist()

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

            results.append({
                "previous_month": previous_month,
                "current_month": current_month,
                "cosine_similarity": round(
                    similarity,
                    4
                ),
                "drift_detected": (
                    similarity < self.threshold
                )
            })

        return pd.DataFrame(results)

    def generate_alerts(
        self,
        similarity_df: pd.DataFrame
    ) -> List[str]:
        """
        Generate drift alerts.
        """

        alerts = []

        drift_rows = similarity_df[
            similarity_df["drift_detected"] == True
        ]

        for _, row in drift_rows.iterrows():

            alert = (
                f"[ALERT] Topic drift detected "
                f"between {row['previous_month']} "
                f"and {row['current_month']} "
                f"(similarity="
                f"{row['cosine_similarity']})"
            )

            alerts.append(alert)

        return alerts

    def detect_drift(
        self,
        df: pd.DataFrame
    ):
        """
        Full topic drift detection pipeline.
        """

        topic_distribution = (
            self.build_topic_distribution(df)
        )

        similarity_df = self.compute_similarity(
            topic_distribution
        )

        alerts = self.generate_alerts(
            similarity_df
        )

        return {
            "topic_distribution": topic_distribution,
            "similarity_scores": similarity_df,
            "alerts": alerts
        }
    
print(TopicDriftDetector)