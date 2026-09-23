"""One-off smoke test for PDF generation from stored pipeline results."""

from io import BytesIO
from pathlib import Path

import pandas as pd

from src.reporting.report_generator import generate_report


def _sample_result(include_sku=True):
    months = ["2025-10", "2025-11", "2025-12"]
    rows = []
    for i, month in enumerate(months):
        rows.append(
            {
                "review": f"Delivery was late {i} <test> & more",
                "date": f"{month}-15",
                "cleaned_review": "delivery late",
                "compound_score": -0.4 if i else 0.1,
                "sentiment_label": "negative" if i else "positive",
                "month": month,
                "topic_id": 0,
                "topic_label": "delivery",
                "sku": "SKU-1" if include_sku else None,
            }
        )
        rows.append(
            {
                "review": "Price is too high",
                "date": f"{month}-20",
                "cleaned_review": "price high",
                "compound_score": -0.2,
                "sentiment_label": "negative",
                "month": month,
                "topic_id": 1,
                "topic_label": "pricing",
                "sku": "SKU-2" if include_sku else None,
            }
        )

    processed = pd.DataFrame(rows)
    processed["date"] = pd.to_datetime(processed["date"])
    if not include_sku:
        processed = processed.drop(columns=["sku"])

    topics_df = pd.DataFrame(
        [
            {
                "topic_id": 0,
                "topic_label": "delivery",
                "keywords": "delivery, late, shipping",
            },
            {
                "topic_id": 1,
                "topic_label": "pricing",
                "keywords": "price, expensive, cost",
            },
        ]
    )

    monthly_sentiment = (
        processed.groupby("month")
        .agg(
            avg_sentiment=("compound_score", "mean"),
            negative_reviews=(
                "sentiment_label",
                lambda x: (x == "negative").sum(),
            ),
            total_reviews=("review", "count"),
        )
        .reset_index()
    )

    topic_frequencies = (
        processed.groupby(["month", "topic_id", "topic_label"])
        .agg(
            frequency=("review", "count"),
            topic_sentiment=("compound_score", "mean"),
            negative_ratio=(
                "sentiment_label",
                lambda x: (x == "negative").mean(),
            ),
        )
        .reset_index()
    )
    topic_frequencies["topic_share"] = 0.5
    topic_frequencies["growth_rate"] = 0.2

    similarity_scores = pd.DataFrame(
        [
            {
                "previous_month": "2025-10",
                "current_month": "2025-11",
                "cosine_similarity": 0.92,
                "topic_drift": 0.08,
                "sentiment_drift": 0.4,
                "volume_drift": 0.1,
                "concept_drift_score": 0.22,
                "concept_drift_detected": False,
            },
            {
                "previous_month": "2025-11",
                "current_month": "2025-12",
                "cosine_similarity": 0.70,
                "topic_drift": 0.30,
                "sentiment_drift": 0.6,
                "volume_drift": 0.2,
                "concept_drift_score": 0.41,
                "concept_drift_detected": True,
            },
        ]
    )

    severity_df = pd.DataFrame(
        [
            {
                "month": "2025-12",
                "topic_label": "delivery",
                "frequency": 2,
                "negative_ratio": 1.0,
                "growth_rate": 0.6,
                "concept_drift_score": 0.41,
                "severity_score": 72.5,
                "severity_level": "High",
                "topic_sentiment": -0.4,
                "recommendation": (
                    "Investigate logistics and delivery partners. "
                    "Prioritize investigation within the current review cycle."
                ),
            },
            {
                "month": "2025-12",
                "topic_label": "pricing",
                "frequency": 1,
                "negative_ratio": 1.0,
                "growth_rate": 0.1,
                "concept_drift_score": 0.41,
                "severity_score": 40.0,
                "severity_level": "Medium",
                "topic_sentiment": -0.2,
                "recommendation": "Review pricing competitiveness.",
            },
        ]
    )

    return {
        "processed_df": processed,
        "topics_df": topics_df,
        "aggregation_results": {
            "monthly_sentiment": monthly_sentiment,
            "topic_frequencies": topic_frequencies,
            "dominant_topics": pd.DataFrame(),
            "sku_topic_frequencies": None,
        },
        "temporal_topics": None,
        "drift_results": {
            "topic_distribution": pd.DataFrame(),
            "similarity_scores": similarity_scores,
            "alerts": [
                "[ALERT] Significant customer feedback drift detected "
                "between 2025-11 and 2025-12 (drift score=0.41)"
            ],
        },
        "severity_df": severity_df.drop(columns=["recommendation"]),
        "recommendation_df": severity_df,
    }


def main():
    out_dir = Path("src/reporting/_smoke_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    with_sku = generate_report(_sample_result(include_sku=True))
    without_sku = generate_report(_sample_result(include_sku=False))
    empty = generate_report(
        {
            "processed_df": pd.DataFrame(),
            "topics_df": pd.DataFrame(),
            "aggregation_results": {},
            "drift_results": {"similarity_scores": pd.DataFrame(), "alerts": []},
            "severity_df": pd.DataFrame(),
            "recommendation_df": pd.DataFrame(),
        }
    )

    (out_dir / "with_sku.pdf").write_bytes(with_sku)
    (out_dir / "without_sku.pdf").write_bytes(without_sku)
    (out_dir / "empty.pdf").write_bytes(empty)

    for name, payload in [
        ("with_sku", with_sku),
        ("without_sku", without_sku),
        ("empty", empty),
    ]:
        assert payload.startswith(b"%PDF"), name
        assert len(payload) > 500, name

    print("PDF smoke tests passed")
    print("with_sku", len(with_sku))
    print("without_sku", len(without_sku))
    print("empty", len(empty))


if __name__ == "__main__":
    main()
