from src.run_pipeline import run_pipeline


if __name__ == "__main__":

    df, results = run_pipeline("data/raw/customer_reviews.csv")

    print(df.head())

    print("\nSentiment Trend")
    print(results["monthly_sentiment"])

    print("\nTopic Frequency")
    print(results["topic_frequencies"])