from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("LOAN_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Loan Default Predictor", layout="wide")

st.title("Loan Default Predictor")
st.markdown(
    "Diese Streamlit-App sendet Kreditmerkmale an die FastAPI und zeigt die "
    "geschätzte Ausfallwahrscheinlichkeit eines Kreditausfalls an."
)


def default_payload() -> dict[str, Any]:
    return {
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


def call_api(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


base = default_payload()

with st.sidebar:
    st.header("API-Verbindung")
    st.code(API_URL)

    if st.button("Health-Check"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=10)
            st.json(response.json())
        except Exception as exc:
            st.error(f"API nicht erreichbar: {exc}")

    with st.expander("Erklärung der Eingabefelder"):
        st.markdown(
            """
**credit_policy**  
Ja, wenn der Kunde die Kreditvergabekriterien von LendingClub erfüllt, sonst Nein.

**purpose**  
Der Zweck des Kredits. Mögliche Werte sind `credit_card`, `debt_consolidation`, `educational`, `major_purchase`, `small_business` und `all_other`.

**int_rate**  
Der Zinssatz des Kredits als Anteil. Ein Zinssatz von 11 % wird zum Beispiel als `0.11` gespeichert.

**installment**  
Die monatliche Rate, die der Kreditnehmer zahlen müsste, wenn der Kredit vergeben wird.

**log_annual_inc**  
Der natürliche Logarithmus des selbst angegebenen Jahreseinkommens.

**dti**  
Debt-to-Income Ratio, also Verhältnis von Schulden zu Jahreseinkommen.

**fico**  
Der FICO-Kredit-Score des Kreditnehmers.

**days_with_cr_line**  
Die Anzahl der Tage, seit denen der Kreditnehmer über eine Kreditlinie verfügt.

**revol_bal**  
Der offene Revolving-Betrag, also z. B. nicht beglichene Kreditkartenschulden am Ende eines Abrechnungszyklus.

**revol_util**  
Die Auslastung der Revolving-Kreditlinie, also genutzter Anteil der verfügbaren Kreditlinie.

**inq_last_6mths**  
Anzahl der Kreditanfragen durch Gläubiger in den letzten 6 Monaten.

**delinq_2yrs**  
Anzahl der Fälle, in denen der Kreditnehmer in den letzten 2 Jahren mehr als 30 Tage mit einer Zahlung im Rückstand war.

**pub_rec**  
Anzahl negativer öffentlicher Einträge, z. B. Insolvenzen, Steuerpfandrechte oder Gerichtsurteile.
"""
        )

st.subheader("Kreditdaten eingeben")
st.caption(
    "Die Eingaben werden an die FastAPI gesendet und dort mit dem registrierten Modell ausgewertet."
)

col1, col2 = st.columns(2)

with col1:
    credit_policy_label = st.selectbox(
        "Kreditvergabekriterien erfüllt",
        ["Ja", "Nein"],
        index=0,
    )
    purpose = st.selectbox(
        "Kreditzweck",
        [
            "all_other",
            "credit_card",
            "debt_consolidation",
            "educational",
            "home_improvement",
            "major_purchase",
            "small_business",
        ],
        index=2,
    )
    int_rate = st.slider(
        "Zinssatz",
        min_value=0.01,
        max_value=0.30,
        value=float(base["int_rate"]),
        step=0.005,
    )
    installment = st.number_input(
        "Monatliche Rate",
        min_value=1.0,
        value=float(base["installment"]),
    )
    log_annual_inc = st.number_input(
        "log(Jahreseinkommen)",
        min_value=1.0,
        value=float(base["log_annual_inc"]),
    )
    dti = st.number_input(
        "Debt-to-Income Ratio",
        min_value=0.0,
        value=float(base["dti"]),
    )

with col2:
    fico = st.slider(
        "FICO Score",
        min_value=300,
        max_value=850,
        value=int(base["fico"]),
    )
    days_with_cr_line = st.number_input(
        "Tage mit Kreditlinie",
        min_value=0.0,
        value=float(base["days_with_cr_line"]),
    )
    revol_bal = st.number_input(
        "Revolving Balance",
        min_value=0.0,
        value=float(base["revol_bal"]),
    )
    revol_util = st.slider(
        "Revolving Utilization",
        min_value=0.0,
        max_value=150.0,
        value=float(base["revol_util"]),
    )
    inq_last_6mths = st.number_input(
        "Kreditanfragen letzte 6 Monate",
        min_value=0,
        value=int(base["inq_last_6mths"]),
    )
    delinq_2yrs = st.number_input(
        "Zahlungsverzüge in 2 Jahren",
        min_value=0,
        value=int(base["delinq_2yrs"]),
    )
    pub_rec = st.number_input(
        "Negative öffentliche Einträge",
        min_value=0,
        value=int(base["pub_rec"]),
    )

credit_policy = 1 if credit_policy_label == "Ja" else 0

payload = {
    "credit_policy": credit_policy,
    "purpose": purpose,
    "int_rate": float(int_rate),
    "installment": float(installment),
    "log_annual_inc": float(log_annual_inc),
    "dti": float(dti),
    "fico": int(fico),
    "days_with_cr_line": float(days_with_cr_line),
    "revol_bal": float(revol_bal),
    "revol_util": float(revol_util),
    "inq_last_6mths": int(inq_last_6mths),
    "delinq_2yrs": int(delinq_2yrs),
    "pub_rec": int(pub_rec),
}

st.subheader("Aktuelle Eingabe")
st.dataframe(pd.DataFrame([payload]), use_container_width=True)

if st.button("Vorhersage berechnen", type="primary"):
    try:
        result = call_api("/predict", payload)
        probability = result.get("default_risk_probability")

        if probability is not None:
            st.metric("Ausfallwahrscheinlichkeit", f"{probability:.2%}")

            if probability < 0.30:
                st.success("Geschätztes Risiko: niedrig")
            elif probability < 0.60:
                st.warning("Geschätztes Risiko: mittel")
            else:
                st.error("Geschätztes Risiko: hoch")

        st.write("**Interpretation:**", result["interpretation"])
        st.json(result)

    except Exception as exc:
        st.error(f"Vorhersage fehlgeschlagen: {exc}")