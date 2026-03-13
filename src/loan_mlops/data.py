from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, MLRUNS_DIR

RAW_DATA_FILE = RAW_DIR / "loan_data.csv"
VALIDATED_DATA_FILE = PROCESSED_DIR / "loan_data_validated.csv"
PREPROCESSED_DATA_FILE = PROCESSED_DIR / "loan_data_preprocessed.csv"


def ensure_processed_dir() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DIR


def read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def load_raw_data() -> pd.DataFrame:
    return read_csv(RAW_DATA_FILE)


def load_validated_data() -> pd.DataFrame:
    return read_csv(VALIDATED_DATA_FILE)


def load_preprocessed_data() -> pd.DataFrame:
    return read_csv(PREPROCESSED_DATA_FILE)


def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
