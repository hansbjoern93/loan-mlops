from __future__ import annotations

import pandas as pd

from loan_mlops.features import MODEL_COLUMNS, PreprocessingPipeline, preprocess_api_payload


def example_raw_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "credit.policy": 1,
                "purpose": "debt_consolidation",
                "int.rate": 0.12,
                "installment": 250.0,
                "log.annual.inc": 10.5,
                "dti": 14.0,
                "fico": 720,
                "days.with.cr.line": 4200.0,
                "revol.bal": 12000.0,
                "revol.util": 35.0,
                "inq.last.6mths": 1,
                "delinq.2yrs": 0,
                "pub.rec": 0,
                "not.fully.paid": 0,
            }
        ]
    )


def test_preprocessing_pipeline_creates_all_model_columns(tmp_path):
    pipeline = PreprocessingPipeline(output_path=str(tmp_path / "processed.csv"))
    transformed = pipeline.transform(example_raw_dataframe())

    for col in MODEL_COLUMNS:
        assert col in transformed.columns

    assert (tmp_path / "processed.csv").exists()


def test_preprocess_api_payload_returns_correct_schema():
    payload = {
        "credit_policy": 1,
        "purpose": "debt_consolidation",
        "int_rate": 0.12,
        "installment": 250.0,
        "log_annual_inc": 10.5,
        "dti": 14.0,
        "fico": 720,
        "days_with_cr_line": 4200.0,
        "revol_bal": 12000.0,
        "revol_util": 35.0,
        "inq_last_6mths": 1,
        "delinq_2yrs": 0,
        "pub_rec": 0,
    }

    df = preprocess_api_payload(payload)

    assert list(df.columns) == MODEL_COLUMNS
    assert df.shape == (1, len(MODEL_COLUMNS))
    assert df.loc[0, "purpose_debt_consolidation"] == 1