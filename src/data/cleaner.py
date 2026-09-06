import pandas as pd
import re


def parse_rating(value):
    """
    Convert values such as:
        'Rated 1 out of 5 stars'
        'Rated 5 out of 5 stars'

    into:
        1
        5
    """

    if pd.isna(value):
        return None

    match = re.search(r"(\d+)", str(value))

    if match:
        rating = int(match.group(1))

        if 1 <= rating <= 5:
            return rating

    return None


def parse_review_count(value):
    """
    Convert reviewer count into an integer when possible.

    Examples:
        '10 reviews' -> 10
        '1 review'   -> 1
        25           -> 25
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    match = re.search(r"(\d[\d,]*)", value)

    if match:
        return int(match.group(1).replace(",", ""))

    return None


def clean_text(value):
    """
    Basic text cleaning without destroying useful language
    needed by VADER and BERTopic.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    # Remove excessive whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def clean_dataset(df):
    """
    Clean and transform the Amazon reviews dataset.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Remove exact duplicate rows
    # --------------------------------------------------

    df = df.drop_duplicates().copy()

    # --------------------------------------------------
    # 2. Parse rating
    # --------------------------------------------------

    df["rating"] = df["Rating"].apply(parse_rating)

    # --------------------------------------------------
    # 3. Parse reviewer count
    # --------------------------------------------------

    df["review_count"] = df["Review Count"].apply(
        parse_review_count
    )

    # --------------------------------------------------
    # 4. Clean country
    # --------------------------------------------------

    df["country"] = (
        df["Country"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------
    # 5. Parse dates
    # --------------------------------------------------

    df["review_date"] = pd.to_datetime(
        df["Review Date"],
        errors="coerce",
        utc=True
    )

    df["experience_date"] = pd.to_datetime(
        df["Date of Experience"],
        errors="coerce"
    )

    # --------------------------------------------------
    # 6. Clean title and review text
    # --------------------------------------------------

    df["review_title"] = df["Review Title"].apply(clean_text)

    df["review_text"] = df["Review Text"].apply(clean_text)

    # --------------------------------------------------
    # 7. Remove rows with no actual review
    # --------------------------------------------------

    df = df[df["review_text"] != ""].copy()

    # --------------------------------------------------
    # 8. Combine title + review
    # --------------------------------------------------

    df["review"] = (
        df["review_title"] + ". " + df["review_text"]
    ).str.strip()

    # --------------------------------------------------
    # 9. Calculate experience delay
    # --------------------------------------------------

    df["experience_delay_days"] = (
        df["review_date"].dt.tz_localize(None)
        - df["experience_date"]
    ).dt.days

    # --------------------------------------------------
    # 10. Add temporal features
    # --------------------------------------------------

    df["review_month"] = (
    df["review_date"]
    .dt.tz_localize(None)
    .dt.to_period("M")
    .astype(str)
    )

    df["review_year"] = df["review_date"].dt.year

    # --------------------------------------------------
    # 11. Keep only useful columns
    # --------------------------------------------------

    cleaned = df[
        [
            "review",
            "review_title",
            "review_text",
            "rating",
            "country",
            "review_date",
            "experience_date",
            "experience_delay_days",
            "review_count",
            "review_month",
            "review_year"
        ]
    ].copy()

    # Reset index
    cleaned.reset_index(drop=True, inplace=True)

    return cleaned