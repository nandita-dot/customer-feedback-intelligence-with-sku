import streamlit as st
import pandas as pd
import plotly.express as px

from src.run_pipeline import run_pipeline
from src.explainability.explainer import InsightExplainer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "Customer Feedback Intelligence System"
)

st.markdown(
    """
### AI-Powered Customer Intelligence

Temporal topic modelling • Sentiment monitoring •
Concept drift detection • Issue severity scoring •
Corrective recommendations
"""
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

st.sidebar.markdown(
    """
Use the controls below to explore customer feedback
intelligence generated from the uploaded dataset.
"""
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Feedback CSV",
    type=["csv"]
)


# =========================================================
# INITIAL STATE
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload a CSV file from the sidebar to begin analysis."
    )

    st.markdown(
        """
### Expected Dataset

Your CSV should contain at least:

- `review`
- `date`

Optional:

- `sku`

### Analysis Pipeline

**Customer Reviews**
→ Text Preprocessing
→ Sentiment Analysis
→ BERTopic
→ Temporal Topic Modelling
→ Concept Drift Detection
→ Severity Impact Scoring
→ Corrective Recommendations
"""
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    raw_df = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read CSV file: {e}"
    )

    st.stop()


# =========================================================
# RAW DATA PREVIEW
# =========================================================

with st.expander(
    "View Raw Dataset",
    expanded=False
):

    st.dataframe(
        raw_df.head(20),
        use_container_width=True
    )

    st.caption(
        f"Dataset contains {len(raw_df):,} records."
    )


# =========================================================
# RUN PIPELINE
# =========================================================

with st.spinner(
    "Running customer intelligence pipeline..."
):

    try:

        results = run_pipeline(
            raw_df
        )

    except Exception as e:

        st.error(
            "An error occurred while running the NLP pipeline."
        )

        st.exception(e)

        st.stop()


# =========================================================
# EXTRACT RESULTS
# =========================================================

df = results["processed_df"]

topics_df = results["topics_df"]

aggregation_results = (
    results["aggregation_results"]
)

sentiment_df = (
    aggregation_results["monthly_sentiment"]
)

topic_frequency_df = (
    aggregation_results["topic_frequencies"]
)

drift_results = (
    results["drift_results"]
)

severity_df = (
    results["severity_df"]
)

recommendation_df = (
    results["recommendation_df"]
)

temporal_topics = (
    results.get("temporal_topics")
)


# =========================================================
# DATA VALIDATION
# =========================================================

required_columns = [
    "date",
    "topic_id",
    "compound_score",
    "sentiment_label"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Required columns missing from processed data: "
        f"{missing_columns}"
    )

    st.stop()


# =========================================================
# DATE PROCESSING
# =========================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date"]
)

df["month"] = (
    df["date"]
    .dt.to_period("M")
    .astype(str)
)


# =========================================================
# TOPIC LABEL MAPPING
# =========================================================

if not topics_df.empty:

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

else:

    df["topic_label"] = "outlier"


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown("---")

st.subheader(
    "Executive Intelligence Overview"
)


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_reviews = len(df)

valid_topics = (
    df.loc[
        df["topic_id"] != -1,
        "topic_id"
    ]
    .nunique()
)

average_sentiment = (
    df["compound_score"]
    .mean()
)

negative_percentage = (
    (
        df["sentiment_label"]
        == "negative"
    )
    .mean()
    * 100
)


# ---------------------------------------------------------
# Latest Drift
# ---------------------------------------------------------

latest_drift = 0

drift_status = "Stable"

if (
    not drift_results["similarity_scores"].empty
):

    latest_drift_row = (
        drift_results["similarity_scores"]
        .iloc[-1]
    )

    latest_drift = (
        latest_drift_row
        .get(
            "concept_drift_score",
            0
        )
    )

    if latest_drift >= 0.60:

        drift_status = "Critical"

    elif latest_drift >= 0.35:

        drift_status = "Elevated"

    else:

        drift_status = "Stable"


# ---------------------------------------------------------
# Highest Severity
# ---------------------------------------------------------

highest_severity = 0

highest_issue = "None"

if not severity_df.empty:

    latest_month = (
        severity_df["month"].max()
    )

    latest_severity_df = (
        severity_df[
            severity_df["month"]
            == latest_month
        ]
    )

    if not latest_severity_df.empty:

        highest_row = (
            latest_severity_df
            .sort_values(
                "severity_score",
                ascending=False
            )
            .iloc[0]
        )

        highest_severity = (
            highest_row["severity_score"]
        )

        highest_issue = (
            highest_row["topic_label"]
        )


# =========================================================
# KPI DISPLAY
# =========================================================

col1, col2, col3, col4, col5 = (
    st.columns(5)
)

with col1:

    st.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )

with col2:

    st.metric(
        "Discovered Topics",
        valid_topics
    )

with col3:

    st.metric(
        "Average Sentiment",
        f"{average_sentiment:.3f}"
    )

with col4:

    st.metric(
        "Negative Reviews",
        f"{negative_percentage:.1f}%"
    )

with col5:

    st.metric(
        "Top Issue Severity",
        f"{highest_severity:.1f}/100"
    )


# =========================================================
# EXECUTIVE ALERT
# =========================================================

if highest_severity >= 75:

    st.error(
        f"🔴 CRITICAL ISSUE: "
        f"{highest_issue} has a severity score of "
        f"{highest_severity:.1f}/100."
    )

elif highest_severity >= 50:

    st.warning(
        f"🟠 HIGH-PRIORITY ISSUE: "
        f"{highest_issue} has a severity score of "
        f"{highest_severity:.1f}/100."
    )

else:

    st.success(
        "No critical customer issue detected."
    )


# =========================================================
# SECTION 1 — SENTIMENT INTELLIGENCE
# =========================================================

st.markdown("---")

st.header(
    "1. Customer Sentiment Intelligence"
)

st.markdown(
    """
This section tracks how customer sentiment changes over
time and identifies periods of increasing dissatisfaction.
"""
)


if not sentiment_df.empty:

    fig_sentiment = px.line(
        sentiment_df,
        x="month",
        y="avg_sentiment",
        markers=True,
        title="Monthly Customer Sentiment"
    )

    fig_sentiment.add_hline(
        y=0,
        line_dash="dash"
    )

    fig_sentiment.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Sentiment",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )


    # -----------------------------------------------------
    # Negative Review Trend
    # -----------------------------------------------------

    if "negative_reviews" in sentiment_df.columns:

        fig_negative = px.bar(
            sentiment_df,
            x="month",
            y="negative_reviews",
            title="Negative Review Volume"
        )

        fig_negative.update_layout(
            xaxis_title="Month",
            yaxis_title="Negative Reviews"
        )

        st.plotly_chart(
            fig_negative,
            use_container_width=True
        )


# =========================================================
# SECTION 2 — DISCOVERED TOPICS
# =========================================================

st.markdown("---")

st.header(
    "2. Customer Discussion Topics"
)

st.markdown(
    """
BERTopic automatically discovers recurring discussion themes
from customer feedback.
"""
)


if not topics_df.empty:

    display_topics = topics_df.copy()

    st.dataframe(
        display_topics,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SECTION 3 — TEMPORAL TOPIC MODELLING
# =========================================================

st.markdown("---")

st.header(
    "3. Temporal Topic Evolution"
)

st.markdown(
    """
This view tracks how customer discussion topics emerge,
grow, decline and persist over time.
"""
)


if temporal_topics is not None:

    temporal_df = temporal_topics.copy()

    # BERTopic normally returns:
    # Topic, Words, Frequency, Timestamp

    if not temporal_df.empty:

        temporal_df = (
            temporal_df.rename(
                columns={
                    "Topic": "topic_id",
                    "Timestamp": "month",
                    "Frequency": "frequency"
                }
            )
        )

        if "topic_id" in temporal_df.columns:

            temporal_df = temporal_df[
                temporal_df["topic_id"] != -1
            ]

            temporal_df["topic_label"] = (
                temporal_df["topic_id"]
                .map(topic_mapping)
                .fillna(
                    temporal_df["topic_id"]
                    .astype(str)
                )
            )

            if not temporal_df.empty:

                fig_temporal = px.line(
                    temporal_df,
                    x="month",
                    y="frequency",
                    color="topic_label",
                    markers=True,
                    title="Topic Evolution Over Time"
                )

                fig_temporal.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Topic Frequency",
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig_temporal,
                    use_container_width=True
                )

            else:

                st.info(
                    "No temporal topic data available."
                )

else:

    st.info(
        "Temporal topic modelling results are unavailable."
    )


# =========================================================
# TOPIC FREQUENCY TREND
# =========================================================

if not topic_frequency_df.empty:

    st.subheader(
        "Topic Frequency Trends"
    )

    fig_topic_frequency = px.line(
        topic_frequency_df,
        x="month",
        y="frequency",
        color="topic_label",
        markers=True,
        title="Customer Topic Frequency"
    )

    fig_topic_frequency.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Reviews",
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_topic_frequency,
        use_container_width=True
    )


# =========================================================
# SECTION 4 — SKU HOTSPOTS
# =========================================================

if "sku" in df.columns:

    st.markdown("---")

    st.header(
        "4. Product / SKU Intelligence"
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
            ),
            negative_reviews=(
                "sentiment_label",
                lambda x:
                (x == "negative").sum()
            )
        )
        .reset_index()
    )

    sku_summary["negative_percentage"] = (
        sku_summary["negative_reviews"]
        /
        sku_summary["total_reviews"]
        * 100
    )

    sku_summary = (
        sku_summary
        .sort_values(
            "avg_sentiment"
        )
    )

    st.dataframe(
        sku_summary,
        use_container_width=True,
        hide_index=True
    )

    fig_sku = px.bar(
        sku_summary,
        x="sku",
        y="avg_sentiment",
        title="Average Sentiment by SKU"
    )

    fig_sku.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_sku,
        use_container_width=True
    )


# =========================================================
# SECTION 5 — CONCEPT DRIFT
# =========================================================

st.markdown("---")

st.header(
    "5. Concept Drift Detection"
)

st.markdown(
    """
Concept drift identifies significant changes in customer
feedback patterns between consecutive time periods.

The system combines:

- Topic distribution drift
- Sentiment drift
- Review-volume drift
"""
)


similarity_df = (
    drift_results["similarity_scores"]
)


if not similarity_df.empty:

    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------

    st.subheader(
        "Topic Distribution Similarity"
    )

    fig_similarity = px.line(
        similarity_df,
        x="current_month",
        y="cosine_similarity",
        markers=True,
        title="Cosine Similarity Between Consecutive Months"
    )

    fig_similarity.add_hline(
        y=0.80,
        line_dash="dash",
        annotation_text="Drift Threshold"
    )

    fig_similarity.update_layout(
        xaxis_title="Month",
        yaxis_title="Cosine Similarity",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig_similarity,
        use_container_width=True
    )


    # -----------------------------------------------------
    # COMBINED DRIFT
    # -----------------------------------------------------

    if "concept_drift_score" in similarity_df.columns:

        st.subheader(
            "Combined Concept Drift Score"
        )

        fig_combined_drift = px.line(
            similarity_df,
            x="current_month",
            y="concept_drift_score",
            markers=True,
            title="Temporal Concept Drift"
        )

        fig_combined_drift.add_hline(
            y=0.35,
            line_dash="dash",
            annotation_text="Drift Detection Threshold"
        )

        fig_combined_drift.update_layout(
            xaxis_title="Month",
            yaxis_title="Concept Drift Score",
            yaxis_range=[0, 1]
        )

        st.plotly_chart(
            fig_combined_drift,
            use_container_width=True
        )


    # -----------------------------------------------------
    # DRIFT TABLE
    # -----------------------------------------------------

    st.subheader(
        "Drift Analysis"
    )

    drift_columns = [
        "previous_month",
        "current_month",
        "cosine_similarity",
        "topic_drift",
        "sentiment_drift",
        "volume_drift",
        "concept_drift_score"
    ]

    available_drift_columns = [
        column
        for column in drift_columns
        if column in similarity_df.columns
    ]

    st.dataframe(
        similarity_df[
            available_drift_columns
        ],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SECTION 6 — DRIFT ALERTS
# =========================================================

st.subheader(
    "Drift Alerts"
)

alerts = (
    drift_results["alerts"]
)

if alerts:

    for alert in alerts:

        st.warning(
            alert
        )

else:

    st.success(
        "No significant concept drift detected."
    )


# =========================================================
# SECTION 7 — SEVERITY IMPACT SCORING
# =========================================================

st.markdown("---")

st.header(
    "6. Customer Issue Severity & Impact"
)

st.markdown(
    """
The Severity Impact Score prioritizes customer issues using
four factors:

**Negative Sentiment + Issue Volume + Growth + Concept Drift**

Higher scores indicate issues that require greater attention.
"""
)


if not severity_df.empty:

    latest_month = (
        severity_df["month"].max()
    )

    latest_severity_df = (
        severity_df[
            severity_df["month"]
            == latest_month
        ]
        .sort_values(
            "severity_score",
            ascending=False
        )
    )

    st.subheader(
        f"Issue Prioritization — {latest_month}"
    )

    severity_columns = [
        "topic_label",
        "severity_score",
        "severity_level",
        "frequency",
        "negative_ratio",
        "growth_rate",
        "concept_drift_score"
    ]

    available_severity_columns = [
        column
        for column in severity_columns
        if column in latest_severity_df.columns
    ]

    st.dataframe(
        latest_severity_df[
            available_severity_columns
        ],
        use_container_width=True,
        hide_index=True
    )


    # -----------------------------------------------------
    # SEVERITY CHART
    # -----------------------------------------------------

    fig_severity = px.bar(
        latest_severity_df,
        x="topic_label",
        y="severity_score",
        color="severity_level",
        title="Customer Issue Severity"
    )

    fig_severity.update_layout(
        xaxis_title="Customer Issue",
        yaxis_title="Severity Impact Score",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )


# =========================================================
# SECTION 8 — CORRECTIVE RECOMMENDATIONS
# =========================================================

st.markdown("---")

st.header(
    "7. Corrective Recommendations"
)

st.markdown(
    """
The recommendation engine converts detected customer
feedback patterns into actionable corrective measures.
"""
)


if not recommendation_df.empty:

    latest_month = (
        recommendation_df["month"].max()
    )

    latest_recommendations = (
        recommendation_df[
            recommendation_df["month"]
            == latest_month
        ]
        .sort_values(
            "severity_score",
            ascending=False
        )
    )

    for _, row in (
        latest_recommendations
        .head(10)
        .iterrows()
    ):

        severity_level = (
            row["severity_level"]
        )

        severity_score = (
            row["severity_score"]
        )

        topic = (
            row["topic_label"]
        )

        recommendation = (
            row["recommendation"]
        )

        # -----------------------------------------------
        # CRITICAL
        # -----------------------------------------------

        if severity_level == "Critical":

            st.error(
                f"🔴 {topic.upper()} — "
                f"{severity_level} — "
                f"{severity_score:.1f}/100"
            )

        # -----------------------------------------------
        # HIGH
        # -----------------------------------------------

        elif severity_level == "High":

            st.warning(
                f"🟠 {topic.upper()} — "
                f"{severity_level} — "
                f"{severity_score:.1f}/100"
            )

        # -----------------------------------------------
        # MEDIUM / LOW
        # -----------------------------------------------

        else:

            st.info(
                f"🟡 {topic.upper()} — "
                f"{severity_level} — "
                f"{severity_score:.1f}/100"
            )

        st.write(
            recommendation
        )

        st.markdown("---")

else:

    st.info(
        "No recommendations generated."
    )


# =========================================================
# SECTION 9 — BUSINESS INSIGHTS
# =========================================================

st.markdown("---")

st.header(
    "8. Explainable Business Insights"
)

explainer = InsightExplainer()


# ---------------------------------------------------------
# Sentiment Insight
# ---------------------------------------------------------

try:

    sentiment_insight = (
        explainer.explain_sentiment(
            sentiment_df
        )
    )

    st.markdown(
        f"### Sentiment Insight\n{sentiment_insight}"
    )

except Exception:

    st.info(
        "Sentiment insight unavailable."
    )


# ---------------------------------------------------------
# Topic Insight
# ---------------------------------------------------------

try:

    topic_insight = (
        explainer.explain_topic_growth(
            topic_frequency_df
        )
    )

    st.markdown(
        f"### Topic Insight\n{topic_insight}"
    )

except Exception:

    st.info(
        "Topic insight unavailable."
    )


# ---------------------------------------------------------
# Drift Insight
# ---------------------------------------------------------

try:

    drift_insight = (
        explainer.explain_drift(
            similarity_df,
            threshold=0.80
        )
    )

    st.markdown(
        f"### Drift Insight\n{drift_insight}"
    )

except Exception:

    st.info(
        "Drift insight unavailable."
    )


# =========================================================
# SECTION 10 — PROCESSED DATA
# =========================================================

st.markdown("---")

st.header(
    "9. Processed Customer Feedback"
)

st.caption(
    "NLP-enriched dataset generated by the intelligence pipeline."
)

st.dataframe(
    df.head(100),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    """
Customer Feedback Intelligence System |
Temporal Topic Modelling • Sentiment Analysis •
Concept Drift Detection • Severity Impact Scoring •
Corrective Recommendations
"""
)

