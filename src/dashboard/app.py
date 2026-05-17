import streamlit as st
import pandas as pd
import plotly.express as px

from src.run_pipeline import run_pipeline


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence System",
    layout="wide"
)

st.title("Customer Feedback Intelligence System")
st.markdown("Topic + Sentiment + Drift + SKU Intelligence Dashboard")


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "Drift Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01
)


# -------------------------------------------------
# UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Customer Reviews CSV",
    type=["csv"]
)


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
if uploaded_file:

    # -----------------------------
    # Load raw CSV
    # -----------------------------
    raw_df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head())

    required_cols = ["review", "date", "sku"]

    missing = [c for c in required_cols if c not in raw_df.columns]

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    # -----------------------------
    # Save temp file for pipeline
    # -----------------------------
    temp_path = "temp_upload.csv"
    raw_df.to_csv(temp_path, index=False)

    # -----------------------------
    # RUN FULL PIPELINE
    # -----------------------------
    df, results = run_pipeline(temp_path)

    st.success("Pipeline executed successfully")

    # -----------------------------
    # SHOW PROCESSED DATA
    # -----------------------------
    st.subheader("Processed Data (Enriched)")
    st.dataframe(df.head())


    # =================================================
    # SENTIMENT TREND
    # =================================================
    st.subheader("Monthly Sentiment Trend")

    fig_sentiment = px.line(
        results["monthly_sentiment"],
        x="month",
        y="avg_sentiment",
        markers=True
    )

    st.plotly_chart(fig_sentiment, use_container_width=True)


    # =================================================
    # TOPIC FREQUENCY
    # =================================================
    st.subheader("Topic Trends Over Time")

    fig_topics = px.line(
        results["topic_frequencies"],
        x="month",
        y="frequency",
        color="topic_id",
        markers=True
    )

    st.plotly_chart(fig_topics, use_container_width=True)

    st.dataframe(results["topic_frequencies"])


    # =================================================
    # SKU LEVEL ANALYSIS (IMPORTANT ADDITION)
    # =================================================
    st.subheader("SKU-Level Topic Hotspots")

    st.dataframe(results["sku_topic_frequencies"])

    selected_sku = st.selectbox(
        "Select SKU",
        sorted(df["sku"].unique())
    )

    sku_df = df[df["sku"] == selected_sku]

    st.markdown(f"### Reviews for SKU: `{selected_sku}`")

    st.dataframe(
        sku_df[["review", "topic_id", "compound_score", "date"]]
    )


    # =================================================
    # DRIFT (SAFE HANDLING)
    # =================================================
    st.subheader("Drift Monitoring")

    if "topic_frequencies" in results:

        drift_df = results["topic_frequencies"]

        if len(drift_df["month"].unique()) < 2:
            st.warning("Not enough data for drift detection")
        else:
            st.dataframe(drift_df)

    # =================================================
    # SUMMARY
    # =================================================
    st.subheader("System Summary")

    st.write("Total Reviews:", len(df))
    st.write("Total SKUs:", df["sku"].nunique())
    st.write("Topics Found:", df["topic_id"].nunique())

else:
    st.info("Upload a CSV file to start analysis")