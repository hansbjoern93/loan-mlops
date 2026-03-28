from __future__ import annotations

import logging
import time
from datetime import datetime
from types import SimpleNamespace
from typing import List, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .features import MODEL_COLUMNS, preprocess_api_payload
from .config import DEFAULT_MODEL_NAME, DEFAULT_TRACKING_URI

logger = logging.getLogger("loan_api")
logging.basicConfig(level=logging.INFO)


class LoanFeatures(BaseModel):
    credit_policy: int = Field(..., ge=0, le=1, description="1 = Kreditrichtlinie erfüllt")
    purpose: Literal[
        "all_other",
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
    ]
    int_rate: float = Field(..., ge=0, le=1, description="Zinssatz als Anteil, z. B. 0.12")
    installment: float = Field(..., gt=0)
    log_annual_inc: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)
    fico: int = Field(..., ge=300, le=850)
    days_with_cr_line: float = Field(..., ge=0)
    revol_bal: float = Field(..., ge=0)
    revol_util: float = Field(..., ge=0, le=150)
    inq_last_6mths: int = Field(..., ge=0)
    delinq_2yrs: int = Field(..., ge=0)
    pub_rec: int = Field(..., ge=0)


class BatchLoanRequest(BaseModel):
    loans: List[LoanFeatures]


def load_registered_model(model_name: str = DEFAULT_MODEL_NAME, tracking_uri: str = DEFAULT_TRACKING_URI):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"Kein registriertes Modell mit dem Namen '{model_name}' gefunden.")

    staging_versions = [v for v in versions if v.current_stage == "Staging"]
    if staging_versions:
        selected_version = sorted(staging_versions, key=lambda v: int(v.version), reverse=True)[0]
    else:
        selected_version = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]

    model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{selected_version.version}")
    return model, selected_version


def create_app(model=None, model_version=None, model_name: str = DEFAULT_MODEL_NAME, tracking_uri: str = DEFAULT_TRACKING_URI) -> FastAPI:
    startup_error = None
    if model is None:
        try:
            model, model_version = load_registered_model(model_name=model_name, tracking_uri=tracking_uri)
        except Exception as exc:  # API soll auch ohne registriertes Modell starten können
            startup_error = str(exc)
            logger.warning("API startet ohne geladenes Modell: %s", exc)
            model = None
            model_version = SimpleNamespace(version="not-loaded", run_id="n/a", current_stage="None")
    elif model_version is None:
        model_version = SimpleNamespace(version="local-test", run_id="n/a", current_stage="Test")

    app = FastAPI(
        title="Loan Default Prediction API",
        description="API zur Vorhersage von Kreditausfallrisiken",
        version="1.0.0",
    )

    app.state.model = model
    app.state.model_version = model_version
    app.state.model_name = model_name
    app.state.model_columns = MODEL_COLUMNS
    app.state.startup_error = startup_error

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = round(time.time() - start_time, 4)
        response.headers["X-Process-Time"] = str(duration)
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unerwarteter Fehler in der API: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Unerwarteter Fehler in der API"})

    @app.get("/")
    def root():
        return {
            "message": "Loan Default Prediction API läuft",
            "model_name": app.state.model_name,
            "model_version": str(app.state.model_version.version),
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health")
    def health():
        return {
            "status": "ok" if app.state.model is not None else "degraded",
            "model_loaded": app.state.model is not None,
            "model_name": app.state.model_name,
            "model_version": str(app.state.model_version.version),
            "startup_error": app.state.startup_error,
        }

    @app.post("/predict")
    def predict(loan: LoanFeatures):
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Kein registriertes Modell geladen. Bitte zuerst Notebook 03 ausführen.")
        input_df = preprocess_api_payload(loan)
        prediction = app.state.model.predict(input_df)[0]
        probability = None
        if hasattr(app.state.model, "predict_proba"):
            probability = float(app.state.model.predict_proba(input_df)[0][1])
        return {
            "prediction": int(prediction),
            "default_risk_probability": probability,
            "interpretation": "Zahlungsausfall wahrscheinlich" if int(prediction) == 1 else "Kein Zahlungsausfall wahrscheinlich",
        }

    @app.post("/predict_batch")
    def predict_batch(request: BatchLoanRequest):
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Kein registriertes Modell geladen. Bitte zuerst Notebook 03 ausführen.")
        frames = [preprocess_api_payload(loan) for loan in request.loans]
        batch_df = pd.concat(frames, ignore_index=True)
        predictions = app.state.model.predict(batch_df)
        probabilities = [None] * len(predictions)
        if hasattr(app.state.model, "predict_proba"):
            probabilities = [float(value) for value in app.state.model.predict_proba(batch_df)[:, 1]]
        return {
            "count": len(predictions),
            "results": [
                {
                    "prediction": int(pred),
                    "default_risk_probability": prob,
                    "interpretation": "Zahlungsausfall wahrscheinlich" if int(pred) == 1 else "Kein Zahlungsausfall wahrscheinlich",
                }
                for pred, prob in zip(predictions, probabilities)
            ],
        }

    return app
