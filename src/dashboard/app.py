import streamlit as st
import pandas as pd
import plotly.express as px

from src.preprocessing.text_preprocessor import (
    TextPreprocessor
)

from src.sentiment.sentiment_analyzer import (
    SentimentAnalyzer
)

from src.topics.lda_topic_model import (
    LDATopicModel
)

from src.temporal.temporal_aggregator import (
    TemporalAggregator
)

from src.drift.topic_drift_detector import (
    TopicDriftDetector
)


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("Customer Feedback Intelligence System")

st.markdown(
    "Temporal Topic-Sentiment Analytics and Drift Detection"
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

st.sidebar.header("Configuration")

similarity_threshold = st.sidebar.slider(
    "Topic Drift Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01
)

num_topics = st.sidebar.slider(
    "Number of Topics",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)


# -------------------------------------------------
# CSV Upload
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Customer Reviews CSV",
    type=["csv"]
)


# -------------------------------------------------
# Main Dashboard
# -------------------------------------------------

if uploaded_file is not None:

    # ---------------------------------------------
    # Load Data
    # ---------------------------------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_columns = [
        "review",
        "date"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.stop()

    # ---------------------------------------------
    # Date Processing
    # ---------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    # ---------------------------------------------
    # Text Preprocessing
    # ---------------------------------------------

    st.subheader("Text Preprocessing")

    preprocessor = TextPreprocessor()

    tokenized_reviews = []
    cleaned_sentences = []

    progress_bar = st.progress(0)

    for idx, review in enumerate(df["review"]):

        tokens, cleaned_sentence = (
            preprocessor.preprocess(review)
        )

        tokenized_reviews.append(tokens)

        cleaned_sentences.append(cleaned_sentence)

        progress_bar.progress(
            (idx + 1) / len(df)
        )

    df["cleaned_review"] = cleaned_sentences

    st.success("Preprocessing completed")

    # ---------------------------------------------
    # Sentiment Analysis
    # ---------------------------------------------

    st.subheader("Sentiment Analysis")

    sentiment_analyzer = SentimentAnalyzer()

    compound_scores = []
    sentiment_labels = []

    for review in df["cleaned_review"]:

        result = (
            sentiment_analyzer.analyze_review(
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

    st.success("Sentiment analysis completed")

    # ---------------------------------------------
    # Topic Modeling
    # ---------------------------------------------

    st.subheader("Topic Modeling")

    lda_pipeline = LDATopicModel(
        num_topics=num_topics
    )

    lda_pipeline.fit(tokenized_reviews)

    extracted_topics = (
        lda_pipeline.extract_topics()
    )

    st.success("Topic modeling completed")

    # ---------------------------------------------
    # Display Topics
    # ---------------------------------------------

    st.subheader("Extracted Topics")

    for topic in extracted_topics:

        st.markdown(
            f"""
            **Topic {topic['topic_id']}**

            {topic['words']}
            """
        )

    # ---------------------------------------------
    # Assign Dominant Topics
    # ---------------------------------------------

    assigned_topics = []

    for tokens in tokenized_reviews:

        topic_probs = (
            lda_pipeline.get_document_topics(
                tokens
            )
        )

        dominant_topic = max(
            topic_probs,
            key=lambda x: x[1]
        )[0]

        assigned_topics.append(
            dominant_topic
        )

    df["topic_id"] = assigned_topics

    # ---------------------------------------------
    # Temporal Aggregation
    # ---------------------------------------------

    st.subheader("Temporal Aggregation")

    aggregator = TemporalAggregator()

    aggregation_results = (
        aggregator.aggregate_monthly(df)
    )

    sentiment_df = (
        aggregation_results[
            "monthly_sentiment"
        ]
    )

    topic_frequency_df = (
        aggregation_results[
            "topic_frequencies"
        ]
    )

    dominant_topics_df = (
        aggregation_results[
            "dominant_topics"
        ]
    )

    # ---------------------------------------------
    # Drift Detection
    # ---------------------------------------------

    detector = TopicDriftDetector(
        threshold=similarity_threshold
    )

    drift_results = detector.detect_drift(
        topic_frequency_df
    )

    similarity_df = (
        drift_results[
            "similarity_scores"
        ]
    )

    alerts = drift_results["alerts"]

    # ---------------------------------------------
    # Dashboard Layout
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    # ---------------------------------------------
    # Sentiment Trend Graph
    # ---------------------------------------------

    with col1:

        st.subheader("Monthly Sentiment Trend")

        fig_sentiment = px.line(
            sentiment_df,
            x="month",
            y="avg_sentiment",
            markers=True,
            title="Average Sentiment Over Time"
        )

        st.plotly_chart(
            fig_sentiment,
            use_container_width=True
        )

    # ---------------------------------------------
    # Dominant Topics
    # ---------------------------------------------

    with col2:

        st.subheader("Dominant Topics")

        st.dataframe(
            dominant_topics_df,
            use_container_width=True
        )

    # ---------------------------------------------
    # Topic Trend Graph
    # ---------------------------------------------

    st.subheader("Topic Trends Over Time")

    fig_topics = px.line(
        topic_frequency_df,
        x="month",
        y="frequency",
        color="topic_id",
        markers=True,
        title="Topic Frequency Trends"
    )

    st.plotly_chart(
        fig_topics,
        use_container_width=True
    )

    # ---------------------------------------------
    # Drift Similarity Graph
    # ---------------------------------------------

    st.subheader("Topic Drift Monitoring")

    if not similarity_df.empty:

        fig_drift = px.line(
            similarity_df,
            x="current_month",
            y="cosine_similarity",
            markers=True,
            title="Monthly Topic Similarity"
        )

        st.plotly_chart(
            fig_drift,
            use_container_width=True
        )

        st.dataframe(
            similarity_df,
            use_container_width=True
        )

    else:

        st.info(
            "Not enough monthly data for drift detection."
        )

    # ---------------------------------------------
    # Drift Alerts
    # ---------------------------------------------

    st.subheader("Drift Alerts")

    if alerts:

        for alert in alerts:
            st.warning(alert)

    else:

        st.success(
            "No topic drift detected"
        )

    # ---------------------------------------------
    # Sentiment Distribution
    # ---------------------------------------------

    st.subheader("Sentiment Distribution")

    sentiment_counts = (
        df["sentiment_label"]
        .value_counts()
        .reset_index()
    )

    sentiment_counts.columns = [
        "sentiment",
        "count"
    ]

    fig_sentiment_dist = px.pie(
        sentiment_counts,
        names="sentiment",
        values="count",
        title="Sentiment Distribution"
    )

    st.plotly_chart(
        fig_sentiment_dist,
        use_container_width=True
    )

    # ---------------------------------------------
    # Topic Review Explorer
    # ---------------------------------------------

    st.subheader("Example Reviews by Topic")

    selected_topic = st.selectbox(
        "Select Topic",
        sorted(df["topic_id"].unique())
    )

    topic_reviews = df[
        df["topic_id"] == selected_topic
    ][[
        "review",
        "sentiment_label",
        "compound_score"
    ]].head(10)

    st.dataframe(
        topic_reviews,
        use_container_width=True
    )

    # ---------------------------------------------
    # Final Processed Dataset
    # ---------------------------------------------

    st.subheader("Processed Dataset")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

else:

    st.info(
        "Upload a CSV file to begin analysis."
    )