from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
MLRUNS_DIR = PROJECT_ROOT / 'mlruns'
MLFLOW_DB_FILE = PROJECT_ROOT / 'mlflow.db'

DEFAULT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Loan Default Prediction')
DEFAULT_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'loan_default_predictor')

# For Docker Compose the service name is "mlflow". Outside Docker this can be overridden.
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")