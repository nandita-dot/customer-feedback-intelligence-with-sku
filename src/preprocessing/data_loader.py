import pandas as pd
from pathlib import Path


class ReviewDataLoader:
    """
    Loads and validates customer review datasets.
    Expected columns:
    review, date, sku
    """

    REQUIRED_COLUMNS = ["review", "date", "sku"]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        df = pd.read_csv(self.file_path)

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["review"] = df["review"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()

        df = df[(df["review"] != "") & (df["review"].str.lower() != "nan")]

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        df = df.sort_values("date").reset_index(drop=True)

        return df