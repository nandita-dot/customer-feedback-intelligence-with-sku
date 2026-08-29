import pandas as pd

from src.preprocessing.text_preprocessor import (
    TextPreprocessor
)

from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer
)

from src.topics.bertopic_model import (
    BERTopicModel
)

from src.topics.topic_labeler import (
    TopicLabeler
)

from src.temporal.temporal_aggregator import (
    TemporalAggregator
)

from src.drift.topic_drift_detector import (
    TopicDriftDetector
)

from src.severity.severity_scorer import (
    SeverityScorer
)

from src.recommendation.recommendation_engine import (
    RecommendationEngine
)


def run_pipeline(df):

    # ===================================================
    # VALIDATION
    # ===================================================

    required_columns = [
        "review",
        "date"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ===================================================
    # CLEAN DATA
    # ===================================================

    df = df.copy()

    df["review"] = (
        df["review"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df[
        (df["review"] != "")
    ]

    df = df.dropna(
        subset=["date"]
    )

    df = (
        df.sort_values("date")
        .reset_index(drop=True)
    )

    # ===================================================
    # PREPROCESSING
    # ===================================================

    preprocessor = TextPreprocessor()

    cleaned_reviews = []
    tokenized_reviews = []

    for review in df["review"]:

        tokens, cleaned_sentence = (
            preprocessor.preprocess(
                review
            )
        )

        tokenized_reviews.append(tokens)
        cleaned_reviews.append(
            cleaned_sentence
        )

    df["cleaned_review"] = (
        cleaned_reviews
    )

    # ===================================================
    # SENTIMENT
    # ===================================================

    sentiment_model = (
        SentimentAnalyzer()
    )

    compound_scores = []
    sentiment_labels = []

    for review in df["cleaned_review"]:

        result = (
            sentiment_model
            .analyze_review(review)
        )

        compound_scores.append(
            result["compound_score"]
        )

        sentiment_labels.append(
            result["sentiment_label"]
        )

    df["compound_score"] = (
        compound_scores
    )

    df["sentiment_label"] = (
        sentiment_labels
    )

    # ===================================================
    # MONTH
    # ===================================================

    df["month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ===================================================
    # BERTopic
    # ===================================================

    topic_model = BERTopicModel()

    documents = (
        df["cleaned_review"]
        .tolist()
    )

    topic_model.fit(
        documents
    )

    df["topic_id"] = (
        topic_model
        .get_topics_for_documents(
            documents
        )
    )

    # ===================================================
    # TEMPORAL TOPIC MODELING
    # ===================================================

    temporal_topics = (
        topic_model.topics_over_time(
            documents,
            df["month"].tolist()
        )
    )

    # ===================================================
    # TOPIC LABELS
    # ===================================================

    unique_topics = sorted(
        df["topic_id"].unique()
    )

    topic_rows = []

    for topic_id in unique_topics:

        if topic_id == -1:
            continue

        keywords = (
            topic_model
            .get_topic_keywords(
                topic_id
            )
        )

        label = (
            TopicLabeler
            .generate_label(
                keywords
            )
        )

        topic_rows.append({

            "topic_id":
                topic_id,

            "topic_label":
                label,

            "keywords":
                ", ".join(keywords)
        })

    topics_df = pd.DataFrame(
        topic_rows
    )

    # ===================================================
    # MERGE LABELS
    # ===================================================

    topic_mapping = dict(
        zip(
            topics_df["topic_id"],
            topics_df["topic_label"]
        )
    )

    df["topic_label"] = (
        df["topic_id"]
        .map(topic_mapping)
        .fillna("outlier")
    )

    # ===================================================
    # TEMPORAL AGGREGATION
    # ===================================================

    aggregator = (
        TemporalAggregator()
    )

    aggregation_results = (
        aggregator.aggregate_monthly(
            df
        )
    )

    # ===================================================
    # DRIFT DETECTION
    # ===================================================

    drift_detector = (
        TopicDriftDetector(
            threshold=0.80
        )
    )

    drift_results = (
        drift_detector.detect_drift(
            df
        )
    )

    # ===================================================
    # SEVERITY
    # ===================================================

    severity_scorer = (
        SeverityScorer()
    )

    severity_df = (
        severity_scorer.calculate(
            aggregation_results[
                "topic_frequencies"
            ],
            drift_results[
                "similarity_scores"
            ]
        )
    )

    # ===================================================
    # RECOMMENDATIONS
    # ===================================================

    recommendation_engine = (
        RecommendationEngine()
    )

    recommendation_df = (
        recommendation_engine.generate(
            severity_df
        )
    )

    # ===================================================
    # RETURN
    # ===================================================

    return {

        "processed_df":
            df,

        "topics_df":
            topics_df,

        "aggregation_results":
            aggregation_results,

        "temporal_topics":
            temporal_topics,

        "drift_results":
            drift_results,

        "severity_df":
            severity_df,

        "recommendation_df":
            recommendation_df
    }