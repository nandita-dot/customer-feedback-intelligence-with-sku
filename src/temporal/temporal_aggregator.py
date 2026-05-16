import pandas as pd
from typing import Dict


print("LOADING TEMPORAL AGGREGATOR")

class TemporalAggregator:
    """
    Aggregates sentiment and topic metrics over time.
    """

    def __init__(
        self,
        date_column: str = "date",
        sentiment_column: str = "compound_score",
        topic_column: str = "topic_id"
    ):

        self.date_column = date_column
        self.sentiment_column = sentiment_column
        self.topic_column = topic_column

    def prepare_datetime(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Ensure datetime format.
        """

        df = df.copy()

        df[self.date_column] = pd.to_datetime(
            df[self.date_column],
            errors="coerce"
        )

        df = df.dropna(
            subset=[self.date_column]
        )

        return df

    def aggregate_monthly(
        self,
        df: pd.DataFrame
    ) -> Dict:
        """
        Monthly aggregation:
        - average sentiment
        - dominant topic
        - topic frequencies
        """

        df = self.prepare_datetime(df)

        # -----------------------------
        # Extract monthly period
        # -----------------------------
        df["month"] = (
            df[self.date_column]
            .dt.to_period("M")
            .astype(str)
        )

        # -----------------------------
        # Average sentiment
        # -----------------------------
        sentiment_trend = (
            df.groupby("month")[
                self.sentiment_column
            ]
            .mean()
            .reset_index(name="avg_sentiment")
        )

        # -----------------------------
        # Topic frequencies
        # -----------------------------
        topic_frequencies = (
            df.groupby(
                ["month", self.topic_column]
            )
            .size()
            .reset_index(name="frequency")
        )

        # -----------------------------
        # Dominant topic per month
        # -----------------------------
        dominant_topics = (
            topic_frequencies.loc[
                topic_frequencies.groupby("month")[
                    "frequency"
                ].idxmax()
            ]
            .reset_index(drop=True)
        )

        dominant_topics = dominant_topics.rename(
            columns={
                self.topic_column: "dominant_topic"
            }
        )

        return {
            "monthly_sentiment": sentiment_trend,
            "topic_frequencies": topic_frequencies,
            "dominant_topics": dominant_topics
        }
    
print(TemporalAggregator)