from __future__ import annotations

import pandas as pd

from loan_mlops.features import MODEL_COLUMNS, TARGET_COLUMN
from loan_mlops.train import build_model_description, get_model_candidates, split_training_data


def make_dataset(rows: int = 20) -> pd.DataFrame:
    records = []
    for i in range(rows):
        row = {column: float(i + 1) for column in MODEL_COLUMNS}
        row["fico"] = 600 + (i % 10) * 20
        row[TARGET_COLUMN] = i % 2
        records.append(row)
    return pd.DataFrame(records)


def test_split_training_data_separates_target():
    df = make_dataset()
    X_train, X_test, y_train, y_test = split_training_data(df)

    assert TARGET_COLUMN not in X_train.columns
    assert TARGET_COLUMN not in X_test.columns
    assert len(X_train) + len(X_test) == len(df)


def test_get_model_candidates_contains_expected_models():
    candidates = get_model_candidates(random_state=42)
    model_types = {candidate["model_type"] for candidate in candidates}

    assert {"LogisticRegression", "RandomForest", "GradientBoosting"} <= model_types
    assert len(candidates) == 9


def test_build_model_description_contains_key_information():
    best_result = {
        "model_type": "RandomForest",
        "params": {"n_estimators": 100},
        "metrics": {
            "accuracy": 0.81,
            "precision": 0.71,
            "recall": 0.65,
            "f1_score": 0.68,
            "roc_auc": 0.79,
        },
    }

    description = build_model_description(best_result, version=3)

    assert "Loan Default Prediction Model v3" in description
    assert "RandomForest" in description
    assert "Accuracy: 0.8100" in description