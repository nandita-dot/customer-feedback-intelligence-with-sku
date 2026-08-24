import streamlit as st
import pandas as pd
import plotly.express as px

from src.run_pipeline import run_pipeline
from src.temporal.temporal_aggregator import (
    TemporalAggregator
)

from src.explainability.explainer import (
    InsightExplainer
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)

st.title(
    "Customer Feedback Intelligence System"
)

st.markdown(
    """
AI-powered temporal topic analysis,
sentiment monitoring,
SKU hotspot detection,
and drift analytics.
"""
)

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ---------------------------------------------------
    # LOAD RAW DATA
    # ---------------------------------------------------

    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Raw Dataset")

    st.dataframe(raw_df.head())

    # ---------------------------------------------------
    # RUN NLP PIPELINE
    # ---------------------------------------------------

    with st.spinner(
        "Running NLP intelligence pipeline..."
    ):

        results = run_pipeline(raw_df)

    # ---------------------------------------------------
    # EXTRACT RESULTS
    # ---------------------------------------------------

    df = results["processed_df"]

    topics_df = results["topics_df"]

    drift_results = results["drift_results"]

    # ---------------------------------------------------
    # SAFETY FIXES
    # ---------------------------------------------------

    if "date" not in df.columns:
        st.error("Missing 'date' column")
        st.stop()

    if "topic_id" not in df.columns:
        st.error("Missing 'topic_id' column")
        st.stop()

    if "compound_score" not in df.columns:
        st.error("Missing 'compound_score' column")
        st.stop()

    # ---------------------------------------------------
    # CREATE MONTH COLUMN
    # ---------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    df["month"] = (
        df["date"]
        .dt.to_period("M")
        .astype(str)
    )

    # ---------------------------------------------------
    # FIX TOPIC LABEL ISSUE
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
    # TEMPORAL AGGREGATION
    # ---------------------------------------------------

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

    # ---------------------------------------------------
    # ADD TOPIC LABELS TO TOPIC FREQUENCY DF
    # ---------------------------------------------------

    topic_frequency_df["topic_label"] = (
        topic_frequency_df["topic_id"]
        .map(topic_mapping)
    )

    # ---------------------------------------------------
    # KPI METRICS
    # ---------------------------------------------------

    st.subheader("Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Total Reviews",
            len(df)
        )

    with col2:

        st.metric(
            "Discovered Topics",
            df["topic_id"].nunique()
        )

    with col3:

        avg_sentiment = round(
            df["compound_score"].mean(),
            3
        )

        st.metric(
            "Average Sentiment",
            avg_sentiment
        )

    # ---------------------------------------------------
    # TOPIC TABLE
    # ---------------------------------------------------

    st.subheader(
        "Discovered Topics"
    )

    st.dataframe(topics_df)

    # ---------------------------------------------------
    # SENTIMENT TREND GRAPH
    # ---------------------------------------------------

    st.subheader(
        "Monthly Sentiment Trend"
    )

    fig_sentiment = px.line(
        sentiment_df,
        x="month",
        y="avg_sentiment",
        markers=True
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

    st.info(
        """
This graph shows how customer sentiment changes over time.

Positive movement upward:
customers becoming happier.

Negative movement downward:
growing dissatisfaction.
"""
    )

    # ---------------------------------------------------
    # TOPIC TRENDS
    # ---------------------------------------------------

    st.subheader(
        "Topic Trends Over Time"
    )

    fig_topics = px.line(
        topic_frequency_df,
        x="month",
        y="frequency",
        color="topic_label",
        markers=True
    )

    st.plotly_chart(
        fig_topics,
        use_container_width=True
    )

    st.info(
        """
This graph shows which customer issues
or discussion themes are growing over time.

A sudden spike usually indicates:
- operational problems
- product defects
- pricing complaints
- delivery failures
- support overload
"""
    )

    # ---------------------------------------------------
    # SKU HOTSPOTS
    # ---------------------------------------------------

    if "sku" in df.columns:

        st.subheader(
            "SKU Complaint Hotspots"
        )

        sku_summary = (
            df.groupby("sku")
            .agg(
                avg_sentiment=(
                    "compound_score",
                    "mean"
                ),
                total_reviews=(
                    "review",
                    "count"
                )
            )
            .reset_index()
        )

        sku_summary = (
            sku_summary.sort_values(
                "avg_sentiment"
            )
        )

        st.dataframe(sku_summary)

        fig_sku = px.bar(
            sku_summary,
            x="sku",
            y="avg_sentiment"
        )

        st.plotly_chart(
            fig_sku,
            use_container_width=True
        )

        st.info(
            """
Low sentiment SKUs indicate
problematic products.

This helps companies identify:
- defect-heavy products
- bad suppliers
- shipping issues
- packaging problems
"""
        )

    # ---------------------------------------------------
    # DRIFT DETECTION
    # ---------------------------------------------------

    st.subheader(
        "Topic Drift Detection"
    )

    similarity_df = (
        drift_results[
            "similarity_scores"
        ]
    )

    if not similarity_df.empty:

        fig_drift = px.line(
            similarity_df,
            x="current_month",
            y="cosine_similarity",
            markers=True
        )

        st.plotly_chart(
            fig_drift,
            use_container_width=True
        )

        st.info(
            """
Drift measures how much customer discussion changes over time.

High similarity:
customers discussing the same issues.

Low similarity:
new problems or changing priorities emerging.
"""
        )

    # ---------------------------------------------------
    # DRIFT ALERTS
    # ---------------------------------------------------

    st.subheader(
        "Drift Alerts"
    )

    alerts = drift_results["alerts"]

    if alerts:

        for alert in alerts:

            st.warning(alert)

    else:

        st.success(
            "No major topic drift detected."
        )

    # ---------------------------------------------------
    # EXPLAINABILITY LAYER
    # ---------------------------------------------------

    st.subheader(
        "AI Business Insights"
    )

    explainer = InsightExplainer()

    insight_1 = (
        explainer.explain_sentiment(
            sentiment_df
        )
    )

    insight_2 = (
        explainer.explain_topic_growth(
            topic_frequency_df
        )
    )

    st.markdown(
        f"### Sentiment Insight\n{insight_1}"
    )

    st.markdown(
        f"### Topic Insight\n{insight_2}"
    )

    # ---------------------------------------------------
    # RAW PROCESSED DATA
    # ---------------------------------------------------

    st.subheader(
        "Processed Dataset"
    )

    st.dataframe(df.head(50))

else:

    st.info(
        "Upload a CSV file to begin."
    )