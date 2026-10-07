from pathlib import Path
import unittest

import pandas as pd

from src.step6_inference.feature_pipeline import (
    FEATURES,
    _feature_rows_from_counts,
)
from src.step6_inference.inference import (
    DEFAULT_ARTIFACT_PATH,
    OPERATING_CUTOFF,
    load_production_artifact,
    predict_step6,
)

ROOT = Path(__file__).resolve().parents[1]


class Step6InferenceTests(unittest.TestCase):
    def test_frozen_feature_formulas_and_order(self):
        rows = []
        for month, frequency, negative_count in (
            ("2020-01", 20, 10),
            ("2020-02", 20, 12),
            ("2020-03", 20, 14),
        ):
            rows.append(
                {
                    "month": month,
                    "business_category": "delivery",
                    "frequency": frequency,
                    "negative_count": negative_count,
                }
            )
        counts = pd.DataFrame(rows)
        features = _feature_rows_from_counts(
            counts,
            app="Foodpanda",
            start_month="2020-01",
            prediction_month="2020-03",
        )
        delivery = features.loc[features["business_category"].eq("delivery")].iloc[0]
        self.assertEqual(tuple(column for column in features if column in FEATURES), FEATURES)
        self.assertEqual(delivery["frequency"], 20)
        self.assertAlmostEqual(delivery["prevalence"], 1.0)
        self.assertAlmostEqual(delivery["prevalence_growth"], 0.0)
        self.assertAlmostEqual(delivery["growth_acceleration"], 0.0)
        self.assertAlmostEqual(delivery["negative_ratio"], 0.7)
        self.assertAlmostEqual(delivery["negative_ratio_change"], 0.1)
        self.assertEqual(delivery["persistence"], 3)
        self.assertTrue(delivery["step6_eligible"])

    def test_artifact_and_frozen_test_feature_inference(self):
        self.assertTrue(DEFAULT_ARTIFACT_PATH.is_file())
        artifact = load_production_artifact()
        self.assertEqual(tuple(artifact["feature_names"]), FEATURES)
        self.assertEqual(artifact["operating_cutoff"], 0.21)
        self.assertEqual(artifact["training_row_count"], 614)
        self.assertEqual(artifact["split_value"], "train")
        self.assertEqual(artifact["model"].n_estimators, 100)
        self.assertEqual(artifact["model"].class_weight, "balanced")
        self.assertEqual(artifact["imputer"].strategy, "median")

        frozen_test = pd.read_csv(
            ROOT
            / "research"
            / "audit"
            / "step6"
            / "final_clean"
            / "test_assignments.csv",
            usecols=list(FEATURES),
            nrows=5,
        )
        predictions = predict_step6(artifact, frozen_test)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(predictions["escalation_probability"].between(0, 1).all())
        self.assertTrue(predictions["predicted_escalation"].isin([True, False]).all())

    def test_one_row_treeshap_explanation(self):
        artifact = load_production_artifact()
        frozen_test = pd.read_csv(
            ROOT
            / "research"
            / "audit"
            / "step6"
            / "final_clean"
            / "test_assignments.csv",
            usecols=list(FEATURES),
            nrows=1,
        )
        explanation = predict_step6(artifact, frozen_test, include_shap=True)
        self.assertTrue(all(f"shap_{feature}" in explanation for feature in FEATURES))

    def test_cutoff_constant_is_frozen(self):
        self.assertEqual(OPERATING_CUTOFF, 0.21)


if __name__ == "__main__":
    unittest.main()
