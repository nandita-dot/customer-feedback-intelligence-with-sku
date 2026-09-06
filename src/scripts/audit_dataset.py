import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = Path("data/Amazon_Reviews.csv")

df = pd.read_csv(
    DATA_PATH,
    engine="python",
    on_bad_lines="warn"
)

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# --------------------------------------------------
# 2. Missing values
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isnull().sum()

for column, count in missing.items():
    percentage = (count / len(df)) * 100
    print(f"{column:25} {count:6} ({percentage:.2f}%)")


# --------------------------------------------------
# 3. Duplicate rows
# --------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print(f"Duplicate complete rows : {df.duplicated().sum():,}")

print(
    f"Duplicate review texts  : "
    f"{df['Review Text'].duplicated().sum():,}"
)


# --------------------------------------------------
# 4. Rating analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("RATING DISTRIBUTION")
print("=" * 60)

print(df["Rating"].value_counts(dropna=False).sort_index())


# --------------------------------------------------
# 5. Country analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOP COUNTRIES")
print("=" * 60)

print(
    df["Country"]
    .value_counts(dropna=False)
    .head(15)
)


# --------------------------------------------------
# 6. Date analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATE ANALYSIS")
print("=" * 60)

review_dates = pd.to_datetime(
    df["Review Date"],
    errors="coerce"
)

experience_dates = pd.to_datetime(
    df["Date of Experience"],
    errors="coerce"
)

print(f"Valid review dates      : {review_dates.notna().sum():,}")
print(f"Invalid/missing dates   : {review_dates.isna().sum():,}")

print(f"\nEarliest review date    : {review_dates.min()}")
print(f"Latest review date      : {review_dates.max()}")

print(
    f"\nValid experience dates  : "
    f"{experience_dates.notna().sum():,}"
)

print(
    f"Invalid/missing exp.    : "
    f"{experience_dates.isna().sum():,}"
)


# --------------------------------------------------
# 7. Review text analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("REVIEW TEXT ANALYSIS")
print("=" * 60)

review_text = df["Review Text"].fillna("").astype(str)

text_lengths = review_text.str.len()

print(f"Empty reviews           : {(review_text.str.strip() == '').sum():,}")
print(f"Average length          : {text_lengths.mean():.2f}")
print(f"Median length           : {text_lengths.median():.2f}")
print(f"Minimum length          : {text_lengths.min():,}")
print(f"Maximum length          : {text_lengths.max():,}")


# --------------------------------------------------
# 8. Review title analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("REVIEW TITLE ANALYSIS")
print("=" * 60)

titles = df["Review Title"].fillna("").astype(str)

print(
    f"Empty titles            : "
    f"{(titles.str.strip() == '').sum():,}"
)

print(
    f"Unique titles           : "
    f"{titles.nunique():,}"
)


# --------------------------------------------------
# 9. Review count
# --------------------------------------------------

print("\n" + "=" * 60)
print("REVIEW COUNT")
print("=" * 60)

review_count = pd.to_numeric(
    df["Review Count"],
    errors="coerce"
)

print(
    f"Valid review counts     : "
    f"{review_count.notna().sum():,}"
)

print(
    f"Invalid/missing counts  : "
    f"{review_count.isna().sum():,}"
)

print(
    f"Average reviewer count  : "
    f"{review_count.mean():.2f}"
)


# --------------------------------------------------
# 10. Sample reviews
# --------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE REVIEWS")
print("=" * 60)

sample = df[
    ["Country", "Rating", "Review Title", "Review Text"]
].dropna(subset=["Review Text"]).sample(
    min(5, len(df)),
    random_state=42
)

for index, row in sample.iterrows():

    print("\n--- REVIEW ---")
    print(f"Country : {row['Country']}")
    print(f"Rating  : {row['Rating']}")
    print(f"Title   : {row['Review Title']}")
    print(f"Text    : {row['Review Text'][:500]}")


print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)