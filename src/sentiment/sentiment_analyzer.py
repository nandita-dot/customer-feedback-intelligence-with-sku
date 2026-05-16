from typing import Dict

from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

print("LOADING SENTIMENT ANALYZER")

class SentimentAnalyzer:
    """
    VADER-based sentiment analysis module.
    """

    def __init__(self):

        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment_label(
        self,
        compound_score: float
    ) -> str:
        """
        Convert compound score into sentiment label.
        """

        if compound_score >= 0.05:
            return "positive"

        elif compound_score <= -0.05:
            return "negative"

        return "neutral"

    def analyze_review(
        self,
        review: str
    ) -> Dict:
        """
        Analyze a single review.

        Returns:
            {
                "compound_score": float,
                "sentiment_label": str
            }
        """

        scores = self.analyzer.polarity_scores(review)

        compound_score = scores["compound"]

        sentiment_label = self.get_sentiment_label(
            compound_score
        )

        return {
            "compound_score": compound_score,
            "sentiment_label": sentiment_label
        }
    
print(SentimentAnalyzer)