import logging
import time
from datetime import datetime
from typing import List, Literal

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("loan_api")

# --------------------------------------------------
# MLflow / Modell laden
# --------------------------------------------------
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

REGISTERED_MODEL_NAME = "loan_default_predictor"


def load_staging_model():
    """
    Lädt bevorzugt das Staging-Modell.
    Falls kein Staging-Modell existiert, wird die neueste Version geladen.
    """
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

    if not versions:
        raise RuntimeError(
            f"Kein registriertes Modell mit dem Namen '{REGISTERED_MODEL_NAME}' gefunden."
        )

    staging_versions = [v for v in versions if v.current_stage == "Staging"]

    if staging_versions:
        selected_version = sorted(
            staging_versions, key=lambda v: int(v.version), reverse=True
        )[0]
        logger.info(
            f"Staging-Modell gefunden: {REGISTERED_MODEL_NAME} v{selected_version.version}"
        )
    else:
        selected_version = sorted(
            versions, key=lambda v: int(v.version), reverse=True
        )[0]
        logger.warning(
            f"Kein Staging-Modell gefunden. Nutze stattdessen neueste Version: "
            f"{REGISTERED_MODEL_NAME} v{selected_version.version}"
        )

    loaded_model = mlflow.sklearn.load_model(f"runs:/{selected_version.run_id}/model")
    return loaded_model, selected_version


model, model_version = load_staging_model()

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Loan Default Prediction API",
    description="API zur Vorhersage von Kreditausfallrisiken",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Pydantic Modelle
# --------------------------------------------------
class LoanFeatures(BaseModel):
    credit_policy: int = Field(..., ge=0, le=1, description="1 oder 0")
    purpose: Literal[
        "all_other",
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "major_purchase",
        "small_business",
    ]
    int_rate: float = Field(..., ge=0, le=1, description="z. B. 0.12")
    installment: float = Field(..., gt=0)
    log_annual_inc: float = Field(..., gt=0)
    dti: float = Field(..., ge=0)
    fico: int = Field(..., ge=300, le=850)
    days_with_cr_line: float = Field(..., ge=0)
    revol_bal: float = Field(..., ge=0)
    revol_util: float = Field(..., ge=0)
    inq_last_6mths: int = Field(..., ge=0)
    delinq_2yrs: int = Field(..., ge=0)
    pub_rec: int = Field(..., ge=0)


class BatchLoanRequest(BaseModel):
    loans: List[LoanFeatures]


# --------------------------------------------------
# Erwartete Modellspalten
# --------------------------------------------------
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
    "purpose_credit_card",
    "purpose_debt_consolidation",
    "purpose_educational",
    "purpose_home_improvement",
    "purpose_major_purchase",
    "purpose_small_business",
]


# --------------------------------------------------
# Preprocessing für API Input
# --------------------------------------------------
def preprocess_input(loan: LoanFeatures) -> pd.DataFrame:
    annual_income = np.exp(loan.log_annual_inc)

    row = {
        "credit.policy": loan.credit_policy,
        "int.rate": loan.int_rate,
        "installment": loan.installment,
        "log.annual.inc": loan.log_annual_inc,
        "dti": loan.dti,
        "fico": loan.fico,
        "days.with.cr.line": loan.days_with_cr_line,
        "revol.bal": loan.revol_bal,
        "revol.util": loan.revol_util,
        "inq.last.6mths": loan.inq_last_6mths,
        "delinq.2yrs": loan.delinq_2yrs,
        "pub.rec": loan.pub_rec,
        "debt_burden_ratio": loan.installment * loan.dti / annual_income,
        "dti_to_fico": loan.dti / loan.fico,
        "int_rate_to_fico": loan.int_rate / loan.fico,
        "revol_bal_to_inc": loan.revol_bal / annual_income,
        "purpose_credit_card": 0,
        "purpose_debt_consolidation": 0,
        "purpose_educational": 0,
        "purpose_home_improvement": 0,
        "purpose_major_purchase": 0,
        "purpose_small_business": 0,
    }

    if loan.purpose != "all_other":
        dummy_col = f"purpose_{loan.purpose}"
        if dummy_col in row:
            row[dummy_col] = 1

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


# --------------------------------------------------
# Middleware / Error Handling
# --------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Request gestartet: {request.method} {request.url.path}")

    try:
        response = await call_next(request)
        duration = round(time.time() - start_time, 4)
        logger.info(
            f"Request beendet: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | Dauer: {duration}s"
        )
        response.headers["X-Process-Time"] = str(duration)
        return response

    except Exception as exc:
        duration = round(time.time() - start_time, 4)
        logger.exception(
            f"Unhandled error bei {request.method} {request.url.path} "
            f"nach {duration}s: {exc}"
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Interner Serverfehler"},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Global exception handler: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Unerwarteter Fehler in der API"},
    )


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Loan Default Prediction API läuft",
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(model_version.version),
        "model_stage": model_version.current_stage,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": REGISTERED_MODEL_NAME,
        "model_version": str(model_version.version),
        "model_stage": model_version.current_stage,
    }


@app.post("/predict")
def predict(loan: LoanFeatures):
    try:
        input_df = preprocess_input(loan)

        prediction = model.predict(input_df)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])

        return {
            "prediction": int(prediction),
            "default_risk_probability": probability,
            "interpretation": (
                "Zahlungsausfall wahrscheinlich"
                if int(prediction) == 1
                else "Kein Zahlungsausfall wahrscheinlich"
            ),
        }

    except Exception as exc:
        logger.exception("Fehler bei /predict")
        raise HTTPException(
            status_code=500, detail=f"Vorhersage fehlgeschlagen: {str(exc)}"
        )


@app.post("/predict_batch")
def predict_batch(request: BatchLoanRequest):
    try:
        frames = [preprocess_input(loan) for loan in request.loans]
        batch_df = pd.concat(frames, ignore_index=True)

        predictions = model.predict(batch_df)

        probabilities = [None] * len(predictions)
        if hasattr(model, "predict_proba"):
            probabilities = [float(p) for p in model.predict_proba(batch_df)[:, 1]]

        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(
                {
                    "prediction": int(pred),
                    "default_risk_probability": prob,
                    "interpretation": (
                        "Zahlungsausfall wahrscheinlich"
                        if int(pred) == 1
                        else "Kein Zahlungsausfall wahrscheinlich"
                    ),
                }
            )

        return {
            "count": len(results),
            "results": results,
        }

    except Exception as exc:
        logger.exception("Fehler bei /predict_batch")
        raise HTTPException(
            status_code=500, detail=f"Batch-Vorhersage fehlgeschlagen: {str(exc)}"
        )


# --------------------------------------------------
# Optional: direkter Start mit python main.py
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)