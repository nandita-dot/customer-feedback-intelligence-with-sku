"""
Command-line entry point: run the feedback pipeline on a CSV without
the dashboard.

    python main.py                       # uses dummy_dataset.csv
    python main.py path/to/reviews.csv   # your own file (review, date[, sku])
"""

import sys

import pandas as pd

from src.run_pipeline import run_pipeline

DEFAULT_CSV = "dummy_dataset.csv"


def main(csv_path):

    # run_pipeline expects a DataFrame, not a file path
    df = pd.read_csv(csv_path)

    results = run_pipeline(df)

    aggregation = results["aggregation_results"]

    print("\nProcessed reviews")
    print(results["processed_df"].head())

    print("\nMonthly Sentiment")
    print(aggregation["monthly_sentiment"])

    print("\nTopic Frequency")
    print(aggregation["topic_frequencies"])

    print("\nTop Issues (severity)")
    print(
        results["recommendation_df"][
            ["month", "topic_label", "severity_score", "severity_level"]
        ].head(10)
    )

    for alert in results["drift_results"]["alerts"]:
        print(alert)


if __name__ == "__main__":

    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV)
