import pandas as pd


REQUIRED_COLUMNS = [
    "Reviewer Name",
    "Profile Link",
    "Country",
    "Review Count",
    "Review Date",
    "Rating",
    "Review Title",
    "Review Text",
    "Date of Experience"
]


def validate_columns(df):
    """
    Validate that the dataset contains all expected columns.
    """

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return True


def validate_rating_range(df):
    """
    Validate that ratings are between 1 and 5.
    """

    invalid = df[
        df["rating"].notna() &
        ~df["rating"].between(1, 5)
    ]

    if not invalid.empty:
        raise ValueError(
            f"Found {len(invalid)} ratings outside the range 1-5."
        )

    return True


def validate_clean_dataset(df):
    """
    Run all validation checks on the cleaned dataset.
    """

    validate_rating_range(df)

    if "review" not in df.columns:
        raise ValueError("Clean dataset must contain a 'review' column.")

    if df["review"].isna().any():
        raise ValueError("Clean dataset contains missing reviews.")

    return True