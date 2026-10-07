from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

FEATURES = (
    "frequency",
    "prevalence",
    "prevalence_growth",
    "growth_acceleration",
    "negative_ratio",
    "negative_ratio_change",
    "persistence",
)
OPERATING_CUTOFF = 0.21
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = _ROOT / "models" / "step6" / "rf_inference.joblib"


def load_production_artifact(
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Step 6 production model artifact not found: {path}. "
            "Build it with `python -m src.step6_inference.build_artifact`."
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Step 6 artifact must contain a metadata/model bundle.")
    if tuple(artifact.get("feature_names", ())) != FEATURES:
        raise ValueError("Step 6 artifact feature order does not match the frozen order.")
    if artifact.get("operating_cutoff") != OPERATING_CUTOFF:
        raise ValueError("Step 6 artifact cutoff does not match frozen cutoff 0.21.")
    if artifact.get("training_row_count") != 614:
        raise ValueError("Step 6 artifact does not record the frozen 614 training rows.")
    if artifact.get("split_value") != "train":
        raise ValueError("Step 6 artifact training split is not the frozen train split.")
    if artifact.get("imputer") is None or artifact.get("model") is None:
        raise ValueError("Step 6 artifact is missing its fitted imputer or forest.")
    return artifact


def _positive_class_shap_values(
    values: Any,
    classes: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    positive_index = list(classes).index(1)
    if isinstance(values, list):
        result = np.asarray(values[positive_index])
    else:
        result = np.asarray(values)
        if result.ndim == 3:
            if result.shape[-1] == len(classes):
                result = result[:, :, positive_index]
            elif result.shape[0] == len(classes):
                result = result[positive_index]
            else:
                raise ValueError(f"Unsupported TreeSHAP result shape: {result.shape}")
    if result.ndim != 2 or result.shape[1] != feature_count:
        raise ValueError(f"Unsupported positive-class SHAP shape: {result.shape}")
    return result


def predict_step6(
    artifact: dict[str, Any],
    features: pd.DataFrame,
    *,
    include_shap: bool = False,
) -> pd.DataFrame:
    missing = set(FEATURES) - set(features.columns)
    if missing:
        raise ValueError(f"Prediction features are missing columns: {sorted(missing)}")
    if features.empty:
        raise ValueError("At least one feature row is required for prediction.")

    ordered_features = features.loc[:, FEATURES]
    transformed = artifact["imputer"].transform(ordered_features)
    model = artifact["model"]
    probabilities = model.predict_proba(transformed)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted Random Forest has no positive class (1).")
    positive_probability = probabilities[:, classes.index(1)]

    result = features.copy().reset_index(drop=True)
    result["escalation_probability"] = positive_probability
    result["predicted_escalation"] = positive_probability >= OPERATING_CUTOFF
    result["predicted_status"] = np.where(
        result["predicted_escalation"],
        "Predicted escalation",
        "Not predicted escalation",
    )

    if include_shap:
        try:
            import shap
        except ImportError as error:
            raise ImportError(
                "TreeSHAP explanations require the `shap` package."
            ) from error
        explainer = shap.TreeExplainer(model)
        shap_values = _positive_class_shap_values(
            explainer.shap_values(transformed),
            model.classes_,
            len(FEATURES),
        )
        for index, feature in enumerate(FEATURES):
            result[f"shap_{feature}"] = shap_values[:, index]

    return result


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "FEATURES",
    "OPERATING_CUTOFF",
    "load_production_artifact",
    "predict_step6",
]
