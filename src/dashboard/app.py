import streamlit as st
import pandas as pd
import plotly.express as px

from src.run_pipeline import run_pipeline
from src.explainability.explainer import InsightExplainer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1rem;
    }

    .insight-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        background-color: #f7f7f7;
        margin-bottom: 1rem;
    }

    .small-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Customer Feedback Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered analysis of customer sentiment, discussion topics,
    changing feedback patterns, issue severity and corrective actions.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Feedback CSV",
    type=["csv"]
)


# =========================================================
# INITIAL STATE
# =========================================================

if uploaded_file is None:

    st.info(
        "Upload a customer feedback CSV file from the sidebar to begin."
    )

    st.markdown(
        """
        ### Expected Dataset

        Required columns:

        - `review`
        - `date`

        Optional:

        - `sku`

        ### Intelligence Pipeline

        **Customer Reviews**
        → Preprocessing
        → Sentiment Analysis
        → BERTopic
        → Temporal Analysis
        → Concept Drift
        → Severity Scoring
        → Recommendations
        """
    )

    st.stop()


# =========================================================
# LOAD DATA
# =========================================================

try:

    raw_df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read CSV: {e}"
    )

    st.stop()


# =========================================================
# RAW DATA
# =========================================================

with st.expander("View Raw Dataset"):

    st.dataframe(
        raw_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"{len(raw_df):,} records loaded."
    )


# =========================================================
# RUN PIPELINE
# =========================================================

with st.spinner(
    "Running customer intelligence pipeline..."
):

    try:

        results = run_pipeline(raw_df)

    except Exception as e:

        st.error(
            "The customer intelligence pipeline could not be completed."
        )

        st.exception(e)

        st.stop()


# =========================================================
# EXTRACT RESULTS
# =========================================================

df = results["processed_df"].copy()

topics_df = results["topics_df"].copy()

aggregation_results = results["aggregation_results"]

sentiment_df = aggregation_results[
    "monthly_sentiment"
].copy()

topic_frequency_df = aggregation_results[
    "topic_frequencies"
].copy()

drift_results = results["drift_results"]

severity_df = results["severity_df"].copy()

recommendation_df = results[
    "recommendation_df"
].copy()


# =========================================================
# VALIDATION
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
        f"Required processed columns are missing: {missing_columns}"
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

topic_mapping = {}

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


# =========================================================
# LATEST MONTH
# =========================================================

latest_month = df["month"].max()

previous_month = None

available_months = sorted(
    df["month"].unique()
)

if len(available_months) >= 2:

    previous_month = available_months[-2]


# =========================================================
# EXECUTIVE METRICS
# =========================================================

total_reviews = len(df)

valid_topics = df.loc[
    df["topic_id"] != -1,
    "topic_id"
].nunique()

average_sentiment = (
    df["compound_score"].mean()
)

negative_percentage = (
    (
        df["sentiment_label"] == "negative"
    ).mean() * 100
)


# =========================================================
# LATEST MONTH SENTIMENT
# =========================================================

latest_df = df[
    df["month"] == latest_month
]

latest_sentiment = (
    latest_df["compound_score"].mean()
)

latest_negative_percentage = (
    (
        latest_df["sentiment_label"] == "negative"
    ).mean() * 100
)


# =========================================================
# SEVERITY
# =========================================================

latest_severity_df = pd.DataFrame()

highest_severity = 0

highest_issue = "No major issue"

if not severity_df.empty:

    latest_severity_df = severity_df[
        severity_df["month"] == latest_month
    ].copy()

    latest_severity_df = (
        latest_severity_df
        .sort_values(
            "severity_score",
            ascending=False
        )
    )

    if not latest_severity_df.empty:

        highest_row = (
            latest_severity_df.iloc[0]
        )

        highest_severity = float(
            highest_row["severity_score"]
        )

        highest_issue = (
            highest_row["topic_label"]
        )


# =========================================================
# DRIFT
# =========================================================

similarity_df = drift_results[
    "similarity_scores"
].copy()

latest_drift = 0

drift_status = "Stable"

if not similarity_df.empty:

    latest_drift_row = (
        similarity_df.iloc[-1]
    )

    latest_drift = float(
        latest_drift_row.get(
            "concept_drift_score",
            0
        )
    )

    if latest_drift >= 0.60:

        drift_status = "Critical Change"

    elif latest_drift >= 0.35:

        drift_status = "Elevated Change"

    else:

        drift_status = "Stable"


# =========================================================
# EXECUTIVE HEADER
# =========================================================

st.markdown("---")

st.markdown(
    "## Executive Intelligence Overview"
)

st.caption(
    f"Current analysis period: **{latest_month}**"
)


# =========================================================
# KPI CARDS
# =========================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Reviews",
        f"{total_reviews:,}"
    )

with col2:

    st.metric(
        "Topics Discovered",
        valid_topics
    )

with col3:

    st.metric(
        "Current Sentiment",
        f"{latest_sentiment:.2f}"
    )

with col4:

    st.metric(
        "Negative Reviews",
        f"{latest_negative_percentage:.1f}%"
    )

with col5:

    st.metric(
        "Highest Issue Severity",
        f"{highest_severity:.1f}/100"
    )


# =========================================================
# EXECUTIVE ALERT
# =========================================================

if highest_severity >= 75:

    st.error(
        f"🔴 **Immediate attention required:** "
        f"{highest_issue} is the highest-priority customer issue "
        f"with a severity score of {highest_severity:.1f}/100."
    )

elif highest_severity >= 50:

    st.warning(
        f"🟠 **Priority issue detected:** "
        f"{highest_issue} currently has the highest severity "
        f"score at {highest_severity:.1f}/100."
    )

else:

    st.success(
        "🟢 No critical customer issue is currently detected."
    )


# =========================================================
# WHAT NEEDS ATTENTION
# =========================================================

st.markdown("---")

st.header(
    "What Needs Attention?"
)

if not recommendation_df.empty:

    attention_row = (
        recommendation_df
        .sort_values(
            "severity_score",
            ascending=False
        )
        .iloc[0]
    )

    attention_topic = (
        str(attention_row["topic_label"])
        .replace("_", " ")
        .title()
    )

    attention_score = float(
        attention_row["severity_score"]
    )

    attention_level = (
        attention_row["severity_level"]
    )

    attention_frequency = (
        attention_row.get(
            "frequency",
            0
        )
    )

    attention_negative_ratio = (
        attention_row.get(
            "negative_ratio",
            0
        )
    )

    attention_growth = (
        attention_row.get(
            "growth_rate",
            0
        )
    )

    if attention_level == "Critical":

        st.error(
            f"🔴 **{attention_topic}** — "
            f"{attention_score:.1f}/100 — "
            f"{attention_level}"
        )

    elif attention_level == "High":

        st.warning(
            f"🟠 **{attention_topic}** — "
            f"{attention_score:.1f}/100 — "
            f"{attention_level}"
        )

    else:

        st.info(
            f"🟡 **{attention_topic}** — "
            f"{attention_score:.1f}/100 — "
            f"{attention_level}"
        )

    st.markdown(
        f"""
**Why is this receiving attention?**

- Severity score: **{attention_score:.1f}/100**
- Reviews in the latest period: **{int(attention_frequency)}**
- Negative review ratio: **{float(attention_negative_ratio) * 100:.1f}%**
- Growth rate: **{float(attention_growth) * 100:.1f}%**

**Recommended action:**  
{attention_row["recommendation"]}
"""
    )

else:

    st.info(
        "No customer issue currently requires attention."
    )

# =========================================================
# SECTION 1 — CUSTOMER HEALTH
# =========================================================

st.markdown("---")

st.markdown(
    "## 1. Customer Sentiment"
)

st.caption(
    "How customer satisfaction is changing over time."
)

if not sentiment_df.empty:

    fig_sentiment = px.line(
        sentiment_df,
        x="month",
        y="avg_sentiment",
        markers=True,
        title="Customer Sentiment Over Time"
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

    if "negative_reviews" in sentiment_df.columns:

        fig_negative = px.line(
            sentiment_df,
            x="month",
            y="negative_reviews",
            markers=True,
            title="Negative Review Volume"
        )

        fig_negative.update_layout(
            xaxis_title="Month",
            yaxis_title="Negative Reviews",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_negative,
            use_container_width=True
        )


# =========================================================
# SENTIMENT INTERPRETATION
# =========================================================

try:

    explainer = InsightExplainer()

    sentiment_insight = (
        explainer.explain_sentiment(
            sentiment_df
        )
    )

    st.info(
        f"💡 {sentiment_insight}"
    )

except Exception:

    pass


# =========================================================
# SECTION 2 — ISSUE PRIORITIZATION
# =========================================================

st.markdown("---")

st.markdown(
    "## 2. Customer Issue Prioritization"
)

st.caption(
    "Issues are ranked using negative sentiment, frequency, growth and overall feedback drift."
)


if not latest_severity_df.empty:

    display_columns = [
        "topic_label",
        "severity_score",
        "severity_level",
        "frequency",
        "negative_ratio",
        "growth_rate"
    ]

    available_columns = [
        c
        for c in display_columns
        if c in latest_severity_df.columns
    ]

    display_df = (
        latest_severity_df[
            available_columns
        ].copy()
    )

    if "negative_ratio" in display_df.columns:

        display_df["negative_ratio"] = (
            display_df["negative_ratio"] * 100
        ).round(1)

    if "growth_rate" in display_df.columns:

        display_df["growth_rate"] = (
            display_df["growth_rate"] * 100
        ).round(1)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    fig_severity = px.bar(
        latest_severity_df.head(10),
        x="severity_score",
        y="topic_label",
        orientation="h",
        title=f"Highest-Priority Customer Issues — {latest_month}"
    )

    fig_severity.update_layout(
        xaxis_title="Severity Impact Score",
        yaxis_title="Customer Issue",
        xaxis_range=[0, 100]
    )

    st.plotly_chart(
        fig_severity,
        use_container_width=True
    )


# =========================================================
# SECTION 3 — WHAT CHANGED?
# =========================================================

st.markdown("---")

st.markdown(
    "## 3. What Changed in Customer Feedback?"
)

st.caption(
    "Detects changes in the overall distribution and nature of customer feedback."
)


if not similarity_df.empty:

    latest_drift_row = similarity_df.iloc[-1]

    drift_score = latest_drift_row.get(
        "concept_drift_score",
        0
    )

    cosine = latest_drift_row.get(
        "cosine_similarity",
        0
    )

    topic_drift = latest_drift_row.get(
        "topic_drift",
        0
    )

    sentiment_drift = latest_drift_row.get(
        "sentiment_drift",
        0
    )

    volume_drift = latest_drift_row.get(
        "volume_drift",
        0
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "Feedback Change",
            f"{drift_score:.2f}"
        )

    with d2:

        st.metric(
            "Topic Change",
            f"{topic_drift:.2f}"
        )

    with d3:

        st.metric(
            "Sentiment Change",
            f"{sentiment_drift:.2f}"
        )

    with d4:

        st.metric(
            "Volume Change",
            f"{volume_drift:.2f}"
        )

    st.write(
        f"Current status: **{drift_status}**"
    )

    fig_drift = px.line(
        similarity_df,
        x="current_month",
        y="concept_drift_score",
        markers=True,
        title="Overall Customer Feedback Change"
    )

    fig_drift.add_hline(
        y=0.35,
        line_dash="dash",
        annotation_text="Change Threshold"
    )

    fig_drift.update_layout(
        xaxis_title="Month",
        yaxis_title="Concept Drift Score",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        fig_drift,
        use_container_width=True
    )

    if drift_score >= 0.35:

        st.warning(
            "⚠️ Customer feedback patterns have changed "
            "significantly compared with the previous period."
        )

    else:

        st.success(
            "Customer feedback patterns remain relatively stable."
        )


# =========================================================
# DRIFT ALERTS
# =========================================================

alerts = drift_results.get(
    "alerts",
    []
)

if alerts:

    with st.expander("View Detected Drift Alerts"):

        for alert in alerts:

            st.warning(alert)


# =========================================================
# SECTION 4 — TOPIC INTELLIGENCE
# =========================================================

st.markdown("---")

st.markdown(
    "## 4. Topic Intelligence"
)

st.caption(
    "Explore individual customer discussion themes instead of displaying every topic simultaneously."
)


if not topic_frequency_df.empty:

    topic_options = sorted(
        topic_frequency_df[
            "topic_label"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    topic_options = [
        topic
        for topic in topic_options
        if topic != "outlier"
    ]

    if topic_options:

        selected_topic = st.selectbox(
            "Select a customer issue/topic",
            topic_options
        )

        selected_topic_df = (
            topic_frequency_df[
                topic_frequency_df[
                    "topic_label"
                ] == selected_topic
            ]
            .sort_values("month")
        )

        c1, c2 = st.columns(2)

        with c1:

            if not selected_topic_df.empty:

                fig_topic_frequency = px.line(
                    selected_topic_df,
                    x="month",
                    y="frequency",
                    markers=True,
                    title=f"{selected_topic}: Review Volume"
                )

                fig_topic_frequency.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Reviews"
                )

                st.plotly_chart(
                    fig_topic_frequency,
                    use_container_width=True
                )

        with c2:

            if (
                not selected_topic_df.empty
                and "topic_sentiment"
                in selected_topic_df.columns
            ):

                fig_topic_sentiment = px.line(
                    selected_topic_df,
                    x="month",
                    y="topic_sentiment",
                    markers=True,
                    title=f"{selected_topic}: Sentiment"
                )

                fig_topic_sentiment.add_hline(
                    y=0,
                    line_dash="dash"
                )

                fig_topic_sentiment.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Sentiment"
                )

                st.plotly_chart(
                    fig_topic_sentiment,
                    use_container_width=True
                )


# =========================================================
# SECTION 5 — SKU INTELLIGENCE
# =========================================================

if "sku" in df.columns:

    st.markdown("---")

    st.markdown(
        "## 5. Product / SKU Intelligence"
    )

    st.caption(
        "Identify products receiving unusually negative customer feedback."
    )

    sku_summary = (
        df.groupby("sku")
        .agg(
            total_reviews=(
                "review",
                "count"
            ),
            avg_sentiment=(
                "compound_score",
                "mean"
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

    sku_display = sku_summary.copy()

    sku_display["avg_sentiment"] = (
        sku_display["avg_sentiment"]
        .round(3)
    )

    sku_display["negative_percentage"] = (
        sku_display["negative_percentage"]
        .round(1)
    )

    st.dataframe(
        sku_display,
        use_container_width=True,
        hide_index=True
    )

    fig_sku = px.bar(
        sku_summary.head(10),
        x="avg_sentiment",
        y="sku",
        orientation="h",
        title="Lowest-Sentiment Products"
    )

    fig_sku.add_vline(
        x=0,
        line_dash="dash"
    )

    fig_sku.update_layout(
        xaxis_title="Average Sentiment",
        yaxis_title="SKU"
    )

    st.plotly_chart(
        fig_sku,
        use_container_width=True
    )


# =========================================================
# SECTION 6 — RECOMMENDATIONS
# =========================================================

st.markdown("---")

st.markdown(
    "## 6. Recommended Corrective Actions"
)

st.caption(
    "Recommendations generated from the highest-priority customer issues."
)


if not recommendation_df.empty:

    latest_recommendations = (
        recommendation_df[
            recommendation_df["month"]
            == latest_month
        ]
        .sort_values(
            "severity_score",
            ascending=False
        )
        .head(10)
    )

    if not latest_recommendations.empty:

        for _, row in (
            latest_recommendations.iterrows()
        ):

            severity_level = row[
                "severity_level"
            ]

            severity_score = row[
                "severity_score"
            ]

            topic = row[
                "topic_label"
            ]

            recommendation = row[
                "recommendation"
            ]

            if severity_level == "Critical":

                st.error(
                    f"🔴 **{topic}** — "
                    f"{severity_score:.1f}/100 — Critical"
                )

            elif severity_level == "High":

                st.warning(
                    f"🟠 **{topic}** — "
                    f"{severity_score:.1f}/100 — High"
                )

            elif severity_level == "Medium":

                st.info(
                    f"🟡 **{topic}** — "
                    f"{severity_score:.1f}/100 — Medium"
                )

            else:

                st.success(
                    f"🟢 **{topic}** — "
                    f"{severity_score:.1f}/100 — Low"
                )

            st.write(
                recommendation
            )


else:

    st.info(
        "No corrective recommendations are available."
    )


# =========================================================
# SECTION 7 — EXPLAINABLE BUSINESS INSIGHTS
# =========================================================

st.markdown("---")

st.markdown(
    "## 7. Explainable Business Insights"
)

explainer = InsightExplainer()


try:

    sentiment_insight = (
        explainer.explain_sentiment(
            sentiment_df
        )
    )

    st.info(
        f"**Sentiment:** {sentiment_insight}"
    )

except Exception:

    pass


try:

    topic_insight = (
        explainer.explain_topic_growth(
            topic_frequency_df
        )
    )

    st.info(
        f"**Topic:** {topic_insight}"
    )

except Exception:

    pass


try:

    drift_insight = (
        explainer.explain_drift(
            similarity_df,
            threshold=0.80
        )
    )

    st.info(
        f"**Change:** {drift_insight}"
    )

except Exception:

    pass


# =========================================================
# SECTION 8 — TOPIC DETAILS
# =========================================================

st.markdown("---")

st.markdown(
    "## 8. Discovered Topic Details"
)

if not topics_df.empty:

    display_topics = topics_df.copy()

    # =====================================================
    # BUSINESS-LEVEL TOPIC VIEW
    # =====================================================

    display_topics["topic_label"] = (
        display_topics["topic_label"]
        .fillna("general_feedback")
        .astype(str)
        .str.replace("_", " ")
        .str.title()
    )

    # Combine BERTopic topics that received the same
    # business label.
    business_topics = (
        display_topics
        .groupby(
            "topic_label",
            as_index=False
        )
        .agg(
            ber_topic_count=(
                "topic_id",
                "nunique"
            ),
            keywords=(
                "keywords",
                lambda x: ", ".join(
                    dict.fromkeys(
                        word.strip()
                        for value in x
                        for word in str(value).split(",")
                    )
                )
            )
        )
    )

    business_topics = (
        business_topics
        .sort_values(
            "ber_topic_count",
            ascending=False
        )
        .reset_index(drop=True)
    )

    business_topics = (
        business_topics.rename(
            columns={
                "topic_label": "Business Issue",
                "ber_topic_count": "Detected Topic Groups",
                "keywords": "Representative Keywords"
            }
        )
    )

    st.dataframe(
        business_topics,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SECTION 9 — PROCESSED DATA
# =========================================================

st.markdown("---")

st.markdown(
    "## 9. Processed Customer Feedback"
)

st.caption(
    "NLP-enriched customer feedback generated by the pipeline."
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
Sentiment Analysis • BERTopic • Temporal Topic Modelling •
Concept Drift Detection • Severity Impact Scoring •
Corrective Recommendations
"""
)