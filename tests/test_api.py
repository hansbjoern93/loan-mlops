from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from loan_mlops.api import create_app


class DummyModel:
    def predict(self, X):
        return np.array([1 if row["fico"] < 650 else 0 for _, row in X.iterrows()])

    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            default_prob = 0.8 if row["fico"] < 650 else 0.2
            probs.append([1 - default_prob, default_prob])
        return np.array(probs)


def make_client(with_model: bool = True) -> TestClient:
    model = DummyModel() if with_model else None
    app = create_app(model=model)
    return TestClient(app)


def sample_payload(fico: int = 720):
    return {
        "credit_policy": 1,
        "purpose": "debt_consolidation",
        "int_rate": 0.12,
        "installment": 250.0,
        "log_annual_inc": 10.5,
        "dti": 14.0,
        "fico": fico,
        "days_with_cr_line": 4200.0,
        "revol_bal": 12000.0,
        "revol_util": 35.0,
        "inq_last_6mths": 1,
        "delinq_2yrs": 0,
        "pub_rec": 0,
    }


def test_root_endpoint_returns_metadata():
    client = make_client()
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "model_name" in data
    assert "model_version" in data


def test_health_endpoint_with_loaded_model():
    client = make_client()
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True


def test_predict_endpoint_low_risk_case():
    client = make_client()
    response = client.post("/predict", json=sample_payload(fico=720))

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0
    assert data["default_risk_probability"] == 0.2


def test_predict_endpoint_high_risk_case():
    client = make_client()
    response = client.post("/predict", json=sample_payload(fico=610))

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["default_risk_probability"] == 0.8


def test_predict_batch_endpoint():
    client = make_client()
    response = client.post(
        "/predict_batch",
        json={"loans": [sample_payload(720), sample_payload(610)]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert [item["prediction"] for item in data["results"]] == [0, 1]


def test_predict_returns_503_if_model_loading_fails(monkeypatch):
    import loan_mlops.api as api_module

    def failing_loader(*args, **kwargs):
        raise RuntimeError("Kein Modell verfügbar")

    monkeypatch.setattr(api_module, "load_registered_model", failing_loader)

    app = api_module.create_app()
    client = TestClient(app)

    response = client.post("/predict", json=sample_payload())

    assert response.status_code == 503