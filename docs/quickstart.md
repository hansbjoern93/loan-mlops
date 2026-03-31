# Quickstart

Diese Anleitung ermöglicht es, das gesamte System in wenigen Schritten zu starten.

---

## Hinweis

Für ein besseres Onboarding und Verständnis der Anwendung sollten die Notebooks 1 - 8 durchgearbeitet werden. Genauere Erklärungen und Informationen zur Anwendung und zum Code liefert die [onboarding.md](onboarding.md).

---

## Voraussetzungen

- Docker und Docker Compose installiert
- Port 5000, 8000 und 8501 sind frei (sonst ggf. ändern)

---

## 1. Repository klonen

```bash
git clone <repository-url>
cd loan-mlops
```

## 2. System starten

``` bash
docker compose up -d --build
```

## 3. Services im Browser öffnen

### Streamlit-Dashboard

Zur Bedienung der Anwendung

`http://localhost:8501`

### Weitere Services

MLflow UI:

`http://localhost:5000`

API (Dokumentation):

`http://localhost:8000/docs`


## 4. Dashboard verwenden

- Kreditdaten im Dashboard eingeben
- Vorhersage berechnen
- Risiko interpretieren

## Hinweise

- Beim ersten Start kann es etwas dauern, bis alle Container bereit sind
- Falls Änderungen am Code vorgenommen wurden, sollte das System neu gebaut werden (Docker-Container neu starten)
