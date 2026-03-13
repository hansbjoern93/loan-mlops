from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import logging
import numpy as np
import pandas as pd

from .data import PREPROCESSED_DATA_FILE, save_dataframe

logger = logging.getLogger(__name__)

TARGET_COLUMN = "not.fully.paid"
CATEGORICAL_COLUMNS = ["purpose"]
PURPOSE_BASELINE = "all_other"
PURPOSE_DUMMY_COLUMNS = [
    "purpose_credit_card",
    "purpose_debt_consolidation",
    "purpose_educational",
    "purpose_home_improvement",
    "purpose_major_purchase",
    "purpose_small_business",
]

MODEL_COLUMNS = [
    "credit.policy",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
    "debt_burden_ratio",
    "dti_to_fico",
    "int_rate_to_fico",
    "revol_bal_to_inc",
    *PURPOSE_DUMMY_COLUMNS,
]

REQUIRED_BASE_COLUMNS = [
    "credit.policy",
    "purpose",
    "int.rate",
    "installment",
    "log.annual.inc",
    "dti",
    "fico",
    "days.with.cr.line",
    "revol.bal",
    "revol.util",
    "inq.last.6mths",
    "delinq.2yrs",
    "pub.rec",
    TARGET_COLUMN,
]


def _check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Spalten: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.dropna()
    cleaned = cleaned.drop_duplicates()
    return cleaned


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_df = df.copy()
    required = [
        "installment",
        "dti",
        "log.annual.inc",
        "fico",
        "int.rate",
        "revol.bal",
    ]
    _check_required_columns(feature_df, required)

    annual_income = np.exp(feature_df["log.annual.inc"])
    feature_df["debt_burden_ratio"] = feature_df["installment"] * feature_df["dti"] / annual_income
    feature_df["dti_to_fico"] = feature_df["dti"] / feature_df["fico"]
    feature_df["int_rate_to_fico"] = feature_df["int.rate"] / feature_df["fico"]
    feature_df["revol_bal_to_inc"] = feature_df["revol.bal"] / annual_income
    return feature_df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.get_dummies(df.copy(), columns=CATEGORICAL_COLUMNS, drop_first=True, dtype=int)
    for col in PURPOSE_DUMMY_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded


def reorder_model_columns(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    ordered = df.copy()
    for col in MODEL_COLUMNS:
        if col not in ordered.columns:
            ordered[col] = 0

    column_order = MODEL_COLUMNS.copy()
    if include_target and TARGET_COLUMN in ordered.columns:
        column_order.append(TARGET_COLUMN)

    remaining = [col for col in ordered.columns if col not in column_order]
    return ordered[column_order + remaining]


@dataclass
class PreprocessingPipeline:
    output_path: str | None = str(PREPROCESSED_DATA_FILE)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starte Datenbereinigung")
        return clean_data(df)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Erstelle neue Features")
        return add_engineered_features(df)

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Kodiere kategoriale Variablen")
        return encode_categorical(df)

    def save_data(self, df: pd.DataFrame, filepath: str | None = None) -> pd.DataFrame:
        path = filepath or self.output_path
        if path is not None:
            save_dataframe(df, path)
            logger.info("Vorverarbeitete Daten gespeichert unter %s", path)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self.clean_data(df)
        transformed = self.engineer_features(transformed)
        transformed = self.encode_categorical(transformed)
        transformed = reorder_model_columns(transformed, include_target=True)
        transformed = self.save_data(transformed)
        return transformed


def preprocess_api_payload(payload: Mapping[str, Any] | Any) -> pd.DataFrame:
    def get_value(name: str) -> Any:
        if isinstance(payload, Mapping):
            return payload[name]
        return getattr(payload, name)

    log_annual_inc = float(get_value("log_annual_inc"))
    annual_income = float(np.exp(log_annual_inc))
    fico = int(get_value("fico"))
    purpose = str(get_value("purpose"))

    row = {
        "credit.policy": int(get_value("credit_policy")),
        "int.rate": float(get_value("int_rate")),
        "installment": float(get_value("installment")),
        "log.annual.inc": log_annual_inc,
        "dti": float(get_value("dti")),
        "fico": fico,
        "days.with.cr.line": float(get_value("days_with_cr_line")),
        "revol.bal": float(get_value("revol_bal")),
        "revol.util": float(get_value("revol_util")),
        "inq.last.6mths": int(get_value("inq_last_6mths")),
        "delinq.2yrs": int(get_value("delinq_2yrs")),
        "pub.rec": int(get_value("pub_rec")),
    }

    row["debt_burden_ratio"] = row["installment"] * row["dti"] / annual_income
    row["dti_to_fico"] = row["dti"] / fico
    row["int_rate_to_fico"] = row["int.rate"] / fico
    row["revol_bal_to_inc"] = row["revol.bal"] / annual_income

    for col in PURPOSE_DUMMY_COLUMNS:
        row[col] = 0

    if purpose != PURPOSE_BASELINE:
        dummy_col = f"purpose_{purpose}"
        if dummy_col in row:
            row[dummy_col] = 1

    return pd.DataFrame([row], columns=MODEL_COLUMNS)
