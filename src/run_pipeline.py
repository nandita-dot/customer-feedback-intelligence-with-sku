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


def run_pipeline(df: pd.DataFrame):

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------

    required_columns = [
        "review",
        "date"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ---------------------------------------------------
    # CLEAN DATA
    # ---------------------------------------------------

    df = df.copy()

    df["review"] = (
        df["review"]
        .astype(str)
        .fillna("")
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    df = df.reset_index(drop=True)

    # ---------------------------------------------------
    # TEXT PREPROCESSING
    # ---------------------------------------------------

    preprocessor = TextPreprocessor()

    cleaned_reviews = []

    tokenized_reviews = []

    for review in df["review"]:

        tokens, cleaned_sentence = (
            preprocessor.preprocess(review)
        )

        tokenized_reviews.append(tokens)

        cleaned_reviews.append(
            cleaned_sentence
        )

    df["cleaned_review"] = cleaned_reviews

    # ---------------------------------------------------
    # SENTIMENT ANALYSIS
    # ---------------------------------------------------

    sentiment_model = SentimentAnalyzer()

    compound_scores = []

    sentiment_labels = []

    for review in df["cleaned_review"]:

        result = (
            sentiment_model.analyze_review(
                review
            )
        )

        compound_scores.append(
            result["compound_score"]
        )

        sentiment_labels.append(
            result["sentiment_label"]
        )

    df["compound_score"] = compound_scores

    df["sentiment_label"] = sentiment_labels

    # ---------------------------------------------------
    # TOPIC MODELING
    # ---------------------------------------------------

    topic_model = BERTopicModel()

    topic_model.fit(
        df["cleaned_review"].tolist()
    )

    topic_ids = (
        topic_model.get_topics_for_documents(
            df["cleaned_review"].tolist()
        )
    )

    df["topic_id"] = topic_ids

    # ---------------------------------------------------
    # TOPIC LABELS
    # ---------------------------------------------------

    unique_topics = sorted(
        df["topic_id"].unique()
    )

    topic_rows = []

    for topic_id in unique_topics:

        if topic_id == -1:
            continue

        keywords = (
            topic_model.get_topic_keywords(
                topic_id
            )
        )

        label = (
            TopicLabeler.generate_label(
                keywords
            )
        )

        topic_rows.append({
            "topic_id": topic_id,
            "topic_label": label,
            "keywords": ", ".join(keywords)
        })

    topics_df = pd.DataFrame(topic_rows)

    # ---------------------------------------------------
    # MERGE LABELS INTO MAIN DF
    # ---------------------------------------------------

    topic_mapping = dict(
        zip(
            topics_df["topic_id"],
            topics_df["topic_label"]
        )
    )

    df["topic_label"] = (
        df["topic_id"]
        .map(topic_mapping)
    )

    # ---------------------------------------------------
    # MONTH COLUMN
    # ---------------------------------------------------

    df["month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------
    # TEMPORAL AGGREGATION
    # ---------------------------------------------------

    aggregator = TemporalAggregator()

    aggregation_results = (
        aggregator.aggregate_monthly(df)
    )

    # ---------------------------------------------------
    # DRIFT DETECTION
    # ---------------------------------------------------

    drift_detector = TopicDriftDetector(
        threshold=0.80
    )

    drift_results = (
        drift_detector.detect_drift(
            aggregation_results[
                "topic_frequencies"
            ]
        )
    )

    # ---------------------------------------------------
    # RETURN EVERYTHING
    # ---------------------------------------------------

    return {

        "processed_df": df,

        "topics_df": topics_df,

        "aggregation_results": (
            aggregation_results
        ),

        "drift_results": drift_results
    }