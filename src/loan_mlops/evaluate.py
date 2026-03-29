from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
try:
    import mlflow
    import mlflow.sklearn
except ModuleNotFoundError:
    mlflow = None
import numpy as np
import seaborn as sns
try:
    from mlflow.tracking import MlflowClient
except ModuleNotFoundError:
    MlflowClient = None
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .train import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI


def load_registered_model(
    model_name: str = DEFAULT_MODEL_NAME,
    tracking_uri: str = DEFAULT_TRACKING_URI,
):
    if mlflow is None or MlflowClient is None:
        raise ModuleNotFoundError("mlflow ist nicht installiert.")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    best_model_version = client.get_latest_versions(model_name, stages=["Staging"])[0]
    best_model = mlflow.sklearn.load_model(
    model_uri=f"models:/loan_default_predictor/{best_model_version.version}")
    return client, best_model_version, best_model


def evaluate_model_comprehensive(model, X_test, y_test, model_name: str = "Modell", artifact_dir: str | Path = "."):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix_path = artifact_dir / "confusion_matrix.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Tatsächlicher Wert")
    plt.xlabel("Vorhergesagter Wert")
    plt.tight_layout()
    plt.savefig(confusion_matrix_path)
    plt.close()

    fpr = tpr = None
    roc_curve_path = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_curve_path = artifact_dir / "roc_curve.png"
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC-Kurve - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_curve_path)
        plt.close()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix_path": str(confusion_matrix_path),
        "roc_curve_path": str(roc_curve_path) if roc_curve_path else None,
    }


def validate_model_cv(best_model, X, y, cv: int = 5):
    model = clone(best_model)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(model, X, y, cv=cv_splitter, scoring=metrics, n_jobs=-1)

    results = {}
    for metric in metrics:
        values = scores[f"test_{metric}"]
        results[metric] = {"mean": float(values.mean()), "std": float(values.std())}
    return results


def create_performance_dashboard(
    evaluation_results: dict[str, Any],
    cv_results: dict[str, Any],
    best_model,
    feature_names,
    output_path: str | Path = "performance_dashboard.png",
):
    output_path = Path(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.heatmap(evaluation_results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
    axes[0, 0].set_title("Confusion Matrix")
    axes[0, 0].set_ylabel("Tatsächlicher Wert")
    axes[0, 0].set_xlabel("Vorhergesagter Wert")

    if evaluation_results["fpr"] is not None and evaluation_results["tpr"] is not None:
        axes[0, 1].plot(evaluation_results["fpr"], evaluation_results["tpr"], label=f"ROC-AUC = {evaluation_results['roc_auc']:.4f}")
        axes[0, 1].plot([0, 1], [0, 1], linestyle="--")
        axes[0, 1].set_title("ROC-Kurve")
        axes[0, 1].set_xlabel("False Positive Rate")
        axes[0, 1].set_ylabel("True Positive Rate")
        axes[0, 1].legend()

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    means = [cv_results[m]["mean"] for m in metric_names]
    stds = [cv_results[m]["std"] for m in metric_names]
    axes[1, 0].bar(metric_names, means, yerr=stds)
    axes[1, 0].set_title("Cross-Validation")
    axes[1, 0].set_ylim(0, 1)

    importances = None
    model_name = None
    if hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
        final_model = best_model.named_steps["model"]
        model_name = type(final_model).__name__
        importances = getattr(final_model, "feature_importances_", None)
        if importances is None and hasattr(final_model, "coef_"):
            importances = np.abs(final_model.coef_[0])
    else:
        model_name = type(best_model).__name__
        importances = getattr(best_model, "feature_importances_", None)

    if importances is not None:
        top_idx = np.argsort(importances)[-10:]
        axes[1, 1].barh(np.array(feature_names)[top_idx], np.array(importances)[top_idx])
        axes[1, 1].set_title(f"Top-Features ({model_name})")
    else:
        axes[1, 1].text(0.5, 0.5, "Keine Feature-Importances verfügbar", ha="center", va="center")
        axes[1, 1].set_title("Feature-Importances")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def update_model_version_registry(client: MlflowClient, model_name: str, model_version, evaluation_results, cv_results, best_model) -> str:
    description = f"""
Loan Default Prediction Model v{model_version.version}

Modelltyp:
- {type(best_model).__name__}

Evaluierung auf Testdaten:
- Accuracy: {evaluation_results['accuracy']:.4f}
- Precision: {evaluation_results['precision']:.4f}
- Recall: {evaluation_results['recall']:.4f}
- F1-Score: {evaluation_results['f1_score']:.4f}
- ROC-AUC: {evaluation_results['roc_auc']:.4f}

Cross-Validation:
- Accuracy: {cv_results['accuracy']['mean']:.4f} (± {cv_results['accuracy']['std']:.4f})
- Precision: {cv_results['precision']['mean']:.4f} (± {cv_results['precision']['std']:.4f})
- Recall: {cv_results['recall']['mean']:.4f} (± {cv_results['recall']['std']:.4f})
- F1: {cv_results['f1']['mean']:.4f} (± {cv_results['f1']['std']:.4f})
- ROC-AUC: {cv_results['roc_auc']['mean']:.4f} (± {cv_results['roc_auc']['std']:.4f})

Aktualisiert am:
- {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}
""".strip()
    client.update_model_version(name=model_name, version=model_version.version, description=description)
    return description
