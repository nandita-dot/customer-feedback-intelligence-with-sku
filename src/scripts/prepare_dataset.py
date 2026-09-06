import sys
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Allow imports from project root
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))


from src.data.validator import (
    validate_columns,
    validate_clean_dataset
)

from src.data.cleaner import clean_dataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = PROJECT_ROOT / "data" / "Amazon_Reviews.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "amazon_reviews_clean.csv"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("AMAZON REVIEWS DATA PREPARATION")
    print("=" * 60)

    print(f"\nReading dataset:")
    print(INPUT_FILE)

    # --------------------------------------------------
    # Read raw CSV
    # --------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        engine="python"
    )

    print(f"\nOriginal rows: {len(df):,}")

    # --------------------------------------------------
    # Validate columns
    # --------------------------------------------------

    validate_columns(df)

    print("Column validation: PASSED")

    # --------------------------------------------------
    # Clean dataset
    # --------------------------------------------------

    cleaned_df = clean_dataset(df)

    print(f"Cleaned rows:  {len(cleaned_df):,}")

    # --------------------------------------------------
    # Validate cleaned dataset
    # --------------------------------------------------

    validate_clean_dataset(cleaned_df)

    print("Clean dataset validation: PASSED")

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Save cleaned dataset
    # --------------------------------------------------

    cleaned_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved to:")

    print(OUTPUT_FILE)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPARATION SUMMARY")
    print("=" * 60)

    print(f"Original rows : {len(df):,}")
    print(f"Cleaned rows  : {len(cleaned_df):,}")
    print(f"Removed rows  : {len(df) - len(cleaned_df):,}")

    print("\nColumns in cleaned dataset:")

    for column in cleaned_df.columns:
        print(f"  - {column}")

    print("\nRating distribution:")

    print(
        cleaned_df["rating"]
        .value_counts()
        .sort_index()
    )

    print("\nCountry distribution:")

    print(
        cleaned_df["country"]
        .value_counts()
        .head(10)
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()