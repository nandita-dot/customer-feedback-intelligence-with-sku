from __future__ import annotations

import pandas as pd

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.topics.topic_labeler import TopicLabeler

FEATURES = (
    "frequency",
    "prevalence",
    "prevalence_growth",
    "growth_acceleration",
    "negative_ratio",
    "negative_ratio_change",
    "persistence",
)
FROZEN_APPS = ("Foodpanda", "Uber Eats", "Zomato")
FROZEN_CATEGORIES = (
    "delivery",
    "pricing",
    "quality",
    "support",
    "payment",
    "product_performance",
    "usability",
    "packaging",
    "general_feedback",
)
MINIMUM_FREQUENCY = 20


def _feature_rows_from_counts(
    counts: pd.DataFrame,
    *,
    app: str,
    start_month: str,
    prediction_month: str,
) -> pd.DataFrame:
    """Apply the frozen Step 4 monthly formulas to category count history."""
    periods = pd.period_range(start_month, prediction_month, freq="M")
    month_totals = counts.groupby("month")["frequency"].sum()
    records = []

    for category in FROZEN_CATEGORIES:
        category_counts = (
            counts.loc[counts["business_category"].eq(category)]
            .set_index("month")
            .reindex(periods.astype(str))
        )
        frequency = category_counts["frequency"].fillna(0).astype("int64")
        negative_count = category_counts["negative_count"].fillna(0).astype("int64")
        denominator = month_totals.reindex(periods.astype(str)).fillna(0)

        prevalence = frequency.div(denominator.where(denominator.ne(0)))
        negative_ratio = negative_count.div(frequency.where(frequency.ne(0)))
        previous_qualifying = (
            frequency.shift(1).ge(MINIMUM_FREQUENCY).fillna(False)
        )
        prevalence_growth = (prevalence / prevalence.shift(1)) - 1
        prevalence_growth = prevalence_growth.where(previous_qualifying)
        growth_acceleration = prevalence_growth.diff()
        negative_ratio_change = (
            negative_ratio - negative_ratio.shift(1)
        ).where(previous_qualifying)

        persistence = []
        run_length = 0
        for is_qualifying in frequency.ge(MINIMUM_FREQUENCY).tolist():
            run_length = run_length + 1 if is_qualifying else 0
            persistence.append(run_length)

        eligible = (
            frequency.ge(MINIMUM_FREQUENCY)
            & previous_qualifying
            & frequency.shift(2).ge(MINIMUM_FREQUENCY).fillna(False)
        )
        row_position = periods.get_loc(pd.Period(prediction_month, freq="M"))
        row = {
            "app": app,
            "business_category": category,
            "month": prediction_month,
            "frequency": int(frequency.iloc[row_position]),
            "prevalence": prevalence.iloc[row_position],
            "prevalence_growth": prevalence_growth.iloc[row_position],
            "growth_acceleration": growth_acceleration.iloc[row_position],
            "negative_ratio": negative_ratio.iloc[row_position],
            "negative_ratio_change": negative_ratio_change.iloc[row_position],
            "persistence": int(persistence[row_position]),
            "step6_eligible": bool(eligible.iloc[row_position]),
        }
        records.append(row)

    result = pd.DataFrame.from_records(records)
    return result.loc[:, ["app", "business_category", "month", *FEATURES, "step6_eligible"]]


def build_step6_features(
    reviews: pd.DataFrame,
    *,
    app: str,
    prediction_month: str,
) -> pd.DataFrame:
    """Build frozen Step 4 features for categories at a historical as-of month.

    Input records after ``prediction_month`` are discarded before category,
    sentiment, or feature construction so future observations cannot enter.
    """
    if app not in FROZEN_APPS:
        raise ValueError(f"app must be one of {FROZEN_APPS}; got {app!r}.")
    missing = {"review", "date"} - set(reviews.columns)
    if missing:
        raise ValueError(f"Uploaded reviews are missing columns: {sorted(missing)}")

    try:
        prediction_period = pd.Period(prediction_month, freq="M")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"prediction_month must be in YYYY-MM format; got {prediction_month!r}."
        ) from error

    work = reviews.loc[:, ["review", "date"]].copy()
    work["review"] = work["review"].fillna("").astype(str).str.strip()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.loc[work["review"].ne("") & work["date"].notna()].copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    work = work.loc[
        work["date"].dt.to_period("M").le(prediction_period)
    ].copy()
    if work.empty:
        raise ValueError(
            f"No valid uploaded reviews are available by {prediction_period}."
        )

    work["business_category"] = work["review"].map(TopicLabeler.generate_label)
    if not work["business_category"].isin(FROZEN_CATEGORIES).all():
        raise ValueError("Frozen Step 4 category assignment produced an unknown category.")

    preprocessor = TextPreprocessor()
    cleaned_texts = [preprocessor.clean_text(text) for text in work["review"]]
    cleaned_reviews = []
    for start in range(0, len(cleaned_texts), 256):
        docs = preprocessor.nlp.pipe(cleaned_texts[start : start + 256], batch_size=256)
        for doc in docs:
            tokens = [
                token.lemma_.strip()
                for token in doc
                if token.lemma_.strip()
                and token.lemma_.strip() not in preprocessor.stop_words
                and not token.lemma_.strip().isspace()
                and len(token.lemma_.strip()) > 1
            ]
            cleaned_reviews.append(" ".join(tokens))

    sentiment_analyzer = SentimentAnalyzer()
    work["is_negative"] = [
        sentiment_analyzer.analyze_review(text)["sentiment_label"] == "negative"
        for text in cleaned_reviews
    ]
    counts = (
        work.groupby(["month", "business_category"], as_index=False, sort=True)
        .agg(
            frequency=("review", "size"),
            negative_count=("is_negative", "sum"),
        )
    )
    counts["month"] = counts["month"].astype(str)

    return _feature_rows_from_counts(
        counts,
        app=app,
        start_month=str(work["date"].dt.to_period("M").min()),
        prediction_month=str(prediction_period),
    )


__all__ = ["FEATURES", "FROZEN_APPS", "build_step6_features"]
