from __future__ import annotations

import datetime as dt
from typing import Any

import requests
import pandas as pd

try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:
    mlflow = None

try:
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    MlflowClient = None

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_EXPERIMENT_NAME, DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI
from .features import TARGET_COLUMN


def check_mlflow_connection(tracking_uri: str, timeout: int = 5) -> None:
    """
    Prüft, ob der MLflow-Server erreichbar ist.
    Dafür wird ein MLflow-API-Endpunkt angesprochen.
    Ein 400-Fehler ist hier akzeptabel, weil er zeigt, dass der Server antwortet.
    """
    check_url = tracking_uri.rstrip("/") + "/api/2.0/mlflow/experiments/search"

    try:
        response = requests.get(check_url, timeout=timeout)

        # 200 = sauber erreichbar
        # 400 = Server antwortet, Request ist nur inhaltlich nicht vollständig
        if response.status_code not in (200, 400):
            raise RuntimeError(
                f"MLflow antwortet unter '{check_url}', aber mit Status {response.status_code}."
            )

    except Exception as exc:
        raise RuntimeError(
            f"MLflow ist unter '{check_url}' nicht erreichbar. "
            "Prüfe, ob der MLflow-Service läuft."
        ) from exc

def setup_mlflow_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> tuple[MlflowClient, Any]:
    if mlflow is None or MlflowClient is None:
        raise ModuleNotFoundError("mlflow ist nicht installiert.")

    check_mlflow_connection(tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)

    mlflow.set_experiment(experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' konnte nicht erstellt oder geladen werden."
        )

    return client, experiment


def split_training_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    X = df.drop(columns=target_column)
    y = df[target_column]
    stratify_values = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def get_model_candidates(random_state: int = 42) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    logreg_params_list = [
        {"C": 0.1, "max_iter": 1000, "class_weight": None, "random_state": random_state},
        {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": random_state},
        {"C": 10.0, "max_iter": 1000, "class_weight": "balanced", "random_state": random_state},
    ]

    for params in logreg_params_list:
        candidates.append(
            {
                "model_type": "LogisticRegression",
                "run_name": "LogisticRegression",
                "params": params,
                "model": Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(**params)),
                    ]
                ),
            }
        )

    rf_params_list = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 2, "random_state": random_state},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "random_state": random_state},
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 10, "random_state": random_state},
    ]

    for params in rf_params_list:
        candidates.append(
            {
                "model_type": "RandomForest",
                "run_name": "RandomForest",
                "params": params,
                "model": RandomForestClassifier(**params),
            }
        )

    gb_params_list = [
        {"n_estimators": 100, "learning_rate": 0.05, "random_state": random_state},
        {"n_estimators": 200, "learning_rate": 0.05, "random_state": random_state},
        {"n_estimators": 100, "learning_rate": 0.1, "random_state": random_state},
    ]

    for params in gb_params_list:
        candidates.append(
            {
                "model_type": "GradientBoosting",
                "run_name": "GradientBoosting",
                "params": params,
                "model": GradientBoostingClassifier(**params),
            }
        )

    return candidates


def train_and_evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    run_name: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mlflow is None:
        raise ModuleNotFoundError("mlflow ist nicht installiert.")

    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": None,
        }

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)

        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model")

        return {
            "run_id": run.info.run_id,
            "model": model,
            "params": params or {},
            "metrics": metrics,
            "run_name": run_name,
        }


def run_model_search(
    X_train,
    X_test,
    y_train,
    y_test,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    results = []

    for candidate in get_model_candidates():
        if verbose:
            print(
                f"Trainiere {candidate['model_type']} "
                f"mit Parametern: {candidate['params']}"
            )

        result = train_and_evaluate_model(
            candidate["model"],
            X_train,
            X_test,
            y_train,
            y_test,
            run_name=candidate["run_name"],
            params=candidate["params"],
        )

        result["model_type"] = candidate["model_type"]
        results.append(result)

    return results


def results_to_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for result in results:
        rows.append(
            {
                "model_type": result["model_type"],
                "run_id": result["run_id"],
                **result["params"],
                **result["metrics"],
            }
        )

    return pd.DataFrame(rows)


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        results,
        key=lambda item: (
            item["metrics"]["roc_auc"]
            if item["metrics"]["roc_auc"] is not None
            else item["metrics"]["f1_score"]
        ),
    )


def build_model_description(best_result: dict[str, Any], version: int | str) -> str:
    metrics = best_result["metrics"]
    roc_auc = metrics["roc_auc"]
    roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "nicht verfügbar"

    return f"""
Loan Default Prediction Model v{version}

- Modelltyp: {best_result['model_type']}
- Training durchgeführt am: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
- Beste Parameter: {best_result['params']}

Performance:
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1-Score: {metrics['f1_score']:.4f}
- ROC-AUC: {roc_auc_text}
""".strip()


def register_best_model(
    best_result: dict[str, Any],
    client: MlflowClient,
    model_name: str = DEFAULT_MODEL_NAME,
):
    if mlflow is None:
        raise ModuleNotFoundError("mlflow ist nicht installiert.")

    model_version = mlflow.register_model(
        f"runs:/{best_result['run_id']}/model",
        model_name,
    )

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    description = build_model_description(best_result, model_version.version)

    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=description,
    )

    return model_version, description