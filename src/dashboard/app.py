import hashlib
import logging
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.run_pipeline import run_pipeline
from src.explainability.explainer import InsightExplainer
from src.reporting.report_generator import generate_report
from src.step6_inference.feature_pipeline import (
    FEATURES,
    FROZEN_APPS,
    build_step6_features,
)
from src.step6_inference.inference import (
    OPERATING_CUTOFF,
    load_production_artifact,
    predict_step6,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    layout="wide"
)


@st.cache_resource
def _cached_step6_artifact():
    return load_production_artifact()


@st.cache_data(show_spinner=False)
def _cached_step6_features(
    reviews: pd.DataFrame,
    source_id: str,
    app: str,
    prediction_month: str,
) -> pd.DataFrame:
    return build_step6_features(
        reviews,
        app=app,
        prediction_month=prediction_month,
    )


def _add_step6_shap_explanations(
    predictions: pd.DataFrame,
    eligible_features: pd.DataFrame,
) -> pd.DataFrame:
    artifact = _cached_step6_artifact()
    transformed = artifact["imputer"].transform(
        eligible_features.loc[:, FEATURES]
    )
    try:
        import shap
    except ImportError as error:
        raise ImportError(
            "TreeSHAP explanations require the `shap` package."
        ) from error

    model = artifact["model"]
    shap_values = shap.TreeExplainer(model).shap_values(transformed)
    positive_index = list(model.classes_).index(1)
    if isinstance(shap_values, list):
        positive_values = np.asarray(shap_values[positive_index])
    else:
        positive_values = np.asarray(shap_values)
        if positive_values.ndim == 3:
            if positive_values.shape[-1] == len(model.classes_):
                positive_values = positive_values[:, :, positive_index]
            elif positive_values.shape[0] == len(model.classes_):
                positive_values = positive_values[positive_index]
    if positive_values.shape != (len(eligible_features), len(FEATURES)):
        raise ValueError(
            f"Unsupported positive-class SHAP shape: {positive_values.shape}"
        )

    explained = predictions.copy()
    for index, feature in enumerate(FEATURES):
        explained[f"shap_{feature}"] = positive_values[:, index]
    return explained


@st.cache_data(show_spinner=False)
def _read_dataset_csv(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
    .main-title {
        font-size: 2.55rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        margin-bottom: 0.35rem;
        color: #132238;
    }
    .subtitle {
        color: #5b6878;
        font-size: 1.08rem;
        margin-bottom: 1.25rem;
    }
    .section-title {
        font-size: 1.45rem;
        font-weight: 650;
        margin-top: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #202b3c !important;
        color: #f8fafc !important;
        border: 1px solid #3b4b61 !important;
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"] *,
    div[data-testid="stMetric"] [data-testid="stMetricDelta"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] *,
    div[data-testid="stMetric"] svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Customer Feedback Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Turn customer feedback into actionable insights about emerging and escalating issues.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# DATASET SELECTION AND ANALYSIS
# =========================================================

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DATASET_PATH = SOURCE_ROOT / "dummy_dataset.csv"
BUILTIN_DATASET_PATHS = {
    "Foodpanda": (
        SOURCE_ROOT
        / "research_data"
        / "extracted"
        / "app-reviews-dataset"
        / "foodpanda_reviews.csv"
    ),
    "Uber Eats": (
        SOURCE_ROOT
        / "research_data"
        / "extracted"
        / "app-reviews-dataset"
        / "ubereats_reviews.csv"
    ),
    "Zomato": (
        SOURCE_ROOT
        / "research_data"
        / "extracted"
        / "app-reviews-dataset"
        / "zomato_reviews.csv"
    ),
}
DATASET_OPTIONS = (
    "Choose a dataset",
    "Sample CSV (repository demo)",
    "Foodpanda",
    "Uber Eats",
    "Zomato",
    "Amazon",
    "Custom CSV",
)

with st.container(border=True):
    st.subheader("Choose customer feedback")
    st.caption("Use the repository sample or upload a CSV with `review` and `date` fields.")
    selected_dataset = st.selectbox(
        "Choose a dataset",
        DATASET_OPTIONS,
        label_visibility="collapsed",
    )

    uploaded_file = None
    input_bytes = None
    source_ready = False
    source_label = selected_dataset

    selected_dataset_path = (
        SAMPLE_DATASET_PATH
        if selected_dataset == "Sample CSV (repository demo)"
        else BUILTIN_DATASET_PATHS.get(selected_dataset)
    )
    if selected_dataset_path is not None:
        selected_dataset_path = selected_dataset_path.resolve()
        if not selected_dataset_path.is_file():
            st.error(
                f"Dataset file is missing: `{selected_dataset_path}`. "
                "No alternate dataset was loaded."
            )
        else:
            try:
                dataset_mtime_ns = selected_dataset_path.stat().st_mtime_ns
                selected_dataframe = _read_dataset_csv(
                    str(selected_dataset_path),
                    dataset_mtime_ns,
                )
                input_bytes = selected_dataset_path.read_bytes()
                missing_fields = {"review", "date"} - set(selected_dataframe.columns)
                parsed_dates = pd.to_datetime(
                    selected_dataframe["date"],
                    errors="coerce",
                ) if "date" in selected_dataframe.columns else pd.Series(dtype="datetime64[ns]")
                valid_dates = parsed_dates.dropna()
                dataset_diagnostics = {
                    "resolved_path": str(selected_dataset_path),
                    "exists": True,
                    "row_count": len(selected_dataframe),
                    "columns": list(selected_dataframe.columns),
                    "parsed_date_min": (
                        str(valid_dates.min()) if not valid_dates.empty else None
                    ),
                    "parsed_date_max": (
                        str(valid_dates.max()) if not valid_dates.empty else None
                    ),
                }
                logging.getLogger(__name__).info(
                    "Selected dataset diagnostics: %s",
                    dataset_diagnostics,
                )
                if missing_fields:
                    st.error(
                        f"{selected_dataset} cannot be loaded by the existing "
                        "pipeline. Required fields missing: "
                        f"{', '.join(sorted(missing_fields))}. "
                        f"File: `{selected_dataset_path}`."
                    )
                elif valid_dates.empty:
                    st.error(
                        f"{selected_dataset} has no parseable values in its `date` "
                        f"column. File: `{selected_dataset_path}`."
                    )
                else:
                    source_ready = True
                    if selected_dataset == "Sample CSV (repository demo)":
                        st.caption(
                            f"Loaded the compatible sample at `{selected_dataset_path.name}`."
                        )
            except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
                st.error(
                    f"Unable to load dataset file `{selected_dataset_path}`: {error}"
                )
    elif selected_dataset == "Custom CSV":
        st.markdown("**Upload your customer feedback CSV**")
        st.markdown("Required columns: `review`, `date`")
        uploaded_file = st.file_uploader(
            "Upload your customer feedback CSV",
            type=["csv"],
            key="custom_feedback_csv",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            input_bytes = uploaded_file.getvalue()
            source_ready = True
            source_label = uploaded_file.name
    elif selected_dataset == "Amazon":
        st.info(
            "The available Amazon CSV uses `Review Text` and `Date of Experience`, "
            "not the pipeline's required `review` and `date` fields. It is not "
            "loaded or silently transformed. Choose Custom CSV with the required fields."
        )

    if input_bytes is not None:
        file_key = hashlib.md5(input_bytes).hexdigest()
        cached_key = st.session_state.get("pipeline_cache_key")
        cached_result = st.session_state.get("pipeline_result")
        already_analyzed = file_key == cached_key and cached_result is not None
        if already_analyzed:
            st.success("This dataset has already been analyzed. Cached results are ready.")
        if st.button(
            "Analyze Feedback",
            type="primary",
            use_container_width=True,
            disabled=not source_ready,
        ):
            try:
                raw_df = pd.read_csv(BytesIO(input_bytes))
            except Exception as error:
                st.error(f"Unable to read CSV: {error}")
                st.stop()

            missing_fields = {"review", "date"} - set(raw_df.columns)
            if missing_fields:
                st.error(
                    "This dataset cannot pass through the existing analysis pipeline. "
                    f"Required fields missing: {', '.join(sorted(missing_fields))}."
                )
                st.stop()

            if file_key == cached_key and cached_result is not None:
                st.session_state["active_source_label"] = source_label
            else:
                with st.spinner("Analyzing customer feedback..."):
                    try:
                        results = run_pipeline(raw_df)
                    except ValueError as error:
                        st.error(str(error))
                        st.stop()
                    except Exception as error:
                        st.error(
                            f"The customer intelligence analysis could not be completed: {error}"
                        )
                        st.stop()

                st.session_state.pipeline_result = results
                st.session_state.pipeline_cache_key = file_key
                st.session_state.pipeline_raw_df = raw_df
                st.session_state.pipeline_run_count = (
                    st.session_state.get("pipeline_run_count", 0) + 1
                )
                st.session_state.active_source_label = source_label
                st.session_state.report_bytes = None
                st.session_state.report_cache_key = None
                st.session_state.report_error = None
            st.rerun()

if selected_dataset == "Choose a dataset":
    st.info("Select a dataset above to begin.")

active_results = st.session_state.get("pipeline_result")
active_file_key = st.session_state.get("pipeline_cache_key")
current_source_key = (
    hashlib.md5(input_bytes).hexdigest() if input_bytes is not None else None
)
if (
    active_results is None
    or current_source_key is None
    or current_source_key != active_file_key
):
    st.stop()

raw_df = st.session_state.pipeline_raw_df
results = active_results
file_key = active_file_key

with st.expander("Preview uploaded dataset"):
    st.dataframe(raw_df.head(20), use_container_width=True, hide_index=True)
    st.caption(f"{len(raw_df):,} records loaded from {st.session_state.get('active_source_label', 'dataset')}.")

st.sidebar.title("Your analysis")
if st.session_state.get("report_cache_key") != file_key:
    try:
        st.session_state.report_bytes = generate_report(results)
        st.session_state.report_cache_key = file_key
        st.session_state.report_error = None
    except Exception as error:
        st.session_state.report_bytes = None
        st.session_state.report_cache_key = file_key
        st.session_state.report_error = str(error)

if st.session_state.get("report_bytes"):
    st.sidebar.download_button(
        label="Download PDF report",
        data=st.session_state.report_bytes,
        file_name="customer_feedback_intelligence_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
elif st.session_state.get("report_error"):
    st.sidebar.error("The report could not be generated.")
    st.sidebar.caption(st.session_state.report_error)


# =========================================================
# EXTRACT RESULTS
# =========================================================

df = results["processed_df"].copy()

analysis_metadata = results.get("analysis_metadata", {})
uploaded_review_count = analysis_metadata.get(
    "uploaded_reviews",
    len(raw_df),
)
reviews_analyzed = analysis_metadata.get(
    "reviews_analyzed",
    len(df),
)

st.info(
    f"**Uploaded reviews:** {uploaded_review_count:,}  \n"
    f"**Reviews analyzed:** {reviews_analyzed:,}"
)

if analysis_metadata.get("sampling_applied", False):
    sample_size = analysis_metadata.get("sample_size", 5000)
    st.warning(
        f"Analysis uses a {sample_size:,}-row sample. "
        "Dashboard metrics, charts, and recommendations represent this "
        "sample, not the full upload."
    )

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
    "## Executive Overview"
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
        "Reviews analyzed",
        f"{total_reviews:,}"
    )

with col2:

    st.metric(
        "Issues detected",
        valid_topics
    )

with col3:

    st.metric(
        "Current Sentiment",
        f"{latest_sentiment:.2f}"
    )

with col4:

    st.metric(
        "Negative feedback",
        f"{latest_negative_percentage:.1f}%"
    )

with col5:

    st.metric(
        "Highest-priority issue",
        str(highest_issue).replace("_", " ").title(),
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
        "No issue meets the dashboard's high-severity threshold "
        "(50/100). Escalation predictions are shown separately below."
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
    "## Customer sentiment over time"
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
    "## What are customers talking about?"
)

st.caption(
    "Issues are ranked using negative sentiment, frequency, growth and overall feedback drift."
)


if not latest_severity_df.empty:

    display_columns = [
        "topic_label",
        "frequency",
        "topic_sentiment",
        "negative_ratio",
        "growth_rate",
        "severity_score",
        "severity_level",
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
    display_df = display_df.rename(
        columns={
            "topic_label": "Issue",
            "frequency": "Mentions",
            "topic_sentiment": "Average sentiment",
            "negative_ratio": "Negative feedback (%)",
            "growth_rate": "Change in mentions (%)",
            "severity_score": "Priority score",
            "severity_level": "Priority",
        }
    )

    if "Negative feedback (%)" in display_df.columns:

        display_df["Negative feedback (%)"] = (
            display_df["Negative feedback (%)"] * 100
        ).round(1)

    if "Change in mentions (%)" in display_df.columns:

        display_df["Change in mentions (%)"] = (
            display_df["Change in mentions (%)"] * 100
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
    "## How are issues changing over time?"
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
# ESCALATION INTELLIGENCE
# =========================================================

st.markdown("---")
st.markdown("## Escalation Intelligence")
st.caption(
    "Identifies issues showing patterns associated with future escalation "
    "based on historical feedback trends. This is a model inference score, "
    "not a calculation of future outcomes."
)

inference_months = sorted(
    pd.to_datetime(raw_df["date"], errors="coerce")
    .dropna()
    .dt.to_period("M")
    .astype(str)
    .unique()
)

if inference_months:
    inference_app = st.selectbox(
        "Which app does this feedback represent?",
        FROZEN_APPS,
        help="Select the app identity used to contextualize this upload.",
    )
    inference_month = st.selectbox(
        "As-of month",
        inference_months,
        index=len(inference_months) - 1,
    )
    include_explanations = st.checkbox(
        "Show why each issue received its score",
        value=False,
    )
    run_inference = st.button(
        "View escalation signals",
        key="run_escalation_inference",
    )

    inference_cache_key = f"{file_key}:{inference_app}:{inference_month}"
    if run_inference:
        try:
            inference_features = _cached_step6_features(
                raw_df.loc[:, ["review", "date"]],
                str(st.session_state.get("active_source_label", "dataset")),
                inference_app,
                inference_month,
            )
            eligible_features = inference_features.loc[
                inference_features["step6_eligible"]
            ].copy()
            if eligible_features.empty:
                st.session_state["escalation_eligible_features"] = eligible_features
                st.session_state["escalation_predictions_key"] = inference_cache_key
                st.session_state["escalation_predictions"] = pd.DataFrame()
            else:
                predictions = predict_step6(
                    _cached_step6_artifact(),
                    eligible_features,
                    include_shap=False,
                )
                st.session_state["escalation_eligible_features"] = eligible_features
                st.session_state["escalation_predictions_key"] = inference_cache_key
                st.session_state["escalation_predictions"] = predictions
        except (FileNotFoundError, ValueError, ImportError) as error:
            st.error(f"Escalation intelligence is unavailable: {error}")
        except Exception as error:
            st.error(f"Unable to display escalation intelligence: {error}")

    if st.session_state.get("escalation_predictions_key") == inference_cache_key:
        predictions = st.session_state.get("escalation_predictions")
        if predictions is None or predictions.empty:
            st.info(
                "Not enough historical data to assess escalation. An issue "
                "must have at least 20 reviews in each of the current and "
                "prior two months."
            )
        else:
            shap_columns = [f"shap_{feature}" for feature in FEATURES]
            if include_explanations and not set(shap_columns).issubset(predictions.columns):
                try:
                    predictions = _add_step6_shap_explanations(
                        predictions,
                        st.session_state["escalation_eligible_features"],
                    )
                    st.session_state["escalation_predictions"] = predictions
                except (FileNotFoundError, ValueError, ImportError) as error:
                    st.error(f"Escalation explanations are unavailable: {error}")
                except Exception as error:
                    st.error(f"Unable to display escalation explanations: {error}")

            predicted_issues = predictions.loc[
                predictions["predicted_escalation"]
            ]
            if predicted_issues.empty:
                st.info("No issues are currently predicted to escalate.")
            else:
                issue_probabilities = ", ".join(
                    f"{str(row['business_category']).replace('_', ' ').title()} "
                    f"({float(row['escalation_probability']):.0%})"
                    for _, row in predicted_issues.iterrows()
                )
                st.warning(f"Predicted escalation: {issue_probabilities}")

            for _, prediction in predictions.sort_values(
                "escalation_probability", ascending=False
            ).iterrows():
                issue_name = str(prediction["business_category"]).replace("_", " ").title()
                probability = float(prediction["escalation_probability"])
                predicted_status = (
                    "Predicted escalation"
                    if prediction["predicted_escalation"]
                    else "Not predicted escalation"
                )
                with st.container(border=True):
                    st.markdown(f"### {issue_name}")
                    issue_columns = st.columns(3)
                    issue_columns[0].metric(
                        "Escalation probability",
                        f"{probability:.0%}",
                    )
                    issue_columns[1].metric("Predicted status", predicted_status)
                    issue_columns[2].metric(
                        "App · Category · Month",
                        f"{inference_app} · {issue_name} · {inference_month}",
                    )

                    if include_explanations and set(shap_columns).issubset(predictions.columns):
                        contribution_columns = {
                            "frequency": "Review volume",
                            "prevalence": "Issue prevalence",
                            "prevalence_growth": "Prevalence growth",
                            "growth_acceleration": "Growth acceleration",
                            "negative_ratio": "Negative feedback",
                            "negative_ratio_change": "Change in negative feedback",
                            "persistence": "Issue persistence",
                        }
                        contributions = pd.DataFrame(
                            [
                                {
                                    "Factor": contribution_columns[feature],
                                    "Contribution": float(prediction[f"shap_{feature}"]),
                                }
                                for feature in contribution_columns
                            ]
                        ).sort_values("Contribution", ascending=False)
                        st.markdown("**Factors contributing to this score**")
                        st.caption(
                            "Positive contributions support the escalation class; "
                            "negative contributions pull the score away from it."
                        )
                        st.bar_chart(
                            contributions.set_index("Factor"),
                            horizontal=True,
                        )
                    with st.expander("Technical details"):
                        feature_values = {
                            feature: prediction[feature]
                            for feature in (
                                "frequency",
                                "prevalence",
                                "prevalence_growth",
                                "growth_acceleration",
                                "negative_ratio",
                                "negative_ratio_change",
                                "persistence",
                            )
                        }
                        st.json(
                            {
                                "model_probability": probability,
                                "operating_cutoff": OPERATING_CUTOFF,
                                "features": feature_values,
                                "shap_contributions": (
                                    {
                                        feature: float(prediction[f"shap_{feature}"])
                                        for feature in feature_values
                                    }
                                    if include_explanations
                                    else None
                                ),
                            }
                        )
            st.caption(
                "Signals use historical review patterns available through the "
                "selected month. They do not state or calculate whether a future "
                "escalation actually occurred."
            )
else:
    st.info("No valid dates are available for escalation inference.")


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
    "## Explore an issue"
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
        "## Product / SKU insights"
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
    "## Recommended actions"
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
    "## Additional business insights"
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
    "## Discovered topic details"
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
    "## Processed review data"
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