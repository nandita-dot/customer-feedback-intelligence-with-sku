from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from src.step6_inference.feature_pipeline import FEATURES
from src.step6_inference.inference import OPERATING_CUTOFF

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_PATH = (
    ROOT / "research" / "audit" / "step6" / "final_clean" / "train_assignments.csv"
)
ARTIFACT_PATH = ROOT / "models" / "step6" / "rf_inference.joblib"
RANDOM_STATE = 20261004
EXPECTED_TRAIN_ROWS = 614
N_TREES = 100


def main() -> None:
    if not TRAINING_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Frozen Step 6 training assignments not found: {TRAINING_DATA_PATH}"
        )
    training = pd.read_csv(TRAINING_DATA_PATH)
    required = {
        "app",
        "business_category",
        "month",
        "escalate",
        "split",
        *FEATURES,
    }
    missing = required - set(training.columns)
    if missing:
        raise ValueError(
            f"Frozen training assignments are missing columns: {sorted(missing)}"
        )
    if len(training) != EXPECTED_TRAIN_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_TRAIN_ROWS} frozen training rows; "
            f"found {len(training)}."
        )
    if not training["split"].eq("train").all():
        raise ValueError("Training assignments contain a non-train split row.")
    if set(training["escalate"].astype(int).unique()) != {0, 1}:
        raise ValueError("Frozen training assignments must contain both label classes.")

    x_train = training.loc[:, FEATURES]
    y_train = training["escalate"].astype(int)
    imputer = SimpleImputer(strategy="median")
    transformed = imputer.fit_transform(x_train)
    forest = RandomForestClassifier(
        n_estimators=N_TREES,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    forest.fit(transformed, y_train)

    source_sha256 = hashlib.sha256(TRAINING_DATA_PATH.read_bytes()).hexdigest()
    artifact = {
        "artifact_version": 1,
        "model": forest,
        "imputer": imputer,
        "feature_names": list(FEATURES),
        "operating_cutoff": OPERATING_CUTOFF,
        "training_row_count": len(training),
        "training_positive_count": int(y_train.sum()),
        "training_data_path": str(TRAINING_DATA_PATH.relative_to(ROOT)),
        "training_data_sha256": source_sha256,
        "split_value": "train",
        "model_metadata": {
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": N_TREES,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "imputer": "sklearn.impute.SimpleImputer(strategy='median')",
            "feature_scaling": False,
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__,
            "training_split": "research/audit/step6/final_clean/train_assignments.csv",
            "training_rows": EXPECTED_TRAIN_ROWS,
            "cutoff_source": (
                "Frozen Step 6 RF CV-selected threshold from "
                "research/audit/step6/final_clean/final_test_metrics.csv"
            ),
        },
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(
        f"Saved Step 6 production inference artifact to {ARTIFACT_PATH}; "
        f"rows={len(training)}, features={len(FEATURES)}, "
        f"cutoff={OPERATING_CUTOFF:.2f}, source_sha256={source_sha256}"
    )


if __name__ == "__main__":
    main()
