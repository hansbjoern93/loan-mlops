# Onboarding – Loan MLOps Projekt

## Ziel dieser Dokumentation

Diese Dokumentation soll den Einstieg in das Projekt erleichtern.  
Sie erklärt den Aufbau der Notebooks, den Datenfluss und die wichtigsten Codeideen so, dass man sich auch mit wenig Python-Erfahrung Schritt für Schritt einarbeiten kann.

Der Fokus liegt dabei nicht nur auf dem Ergebnis, sondern vor allem auf dem Verständnis:

- Was macht jedes Notebook?
- Warum gibt es diesen Schritt?
- Welche Datei wird eingelesen und welche wird erzeugt?
- Welche Funktionen sind wichtig?
- Wie hängen die einzelnen Teile zusammen?
- Wo entstehen typische Fehler?

Die Dokumentation ist so aufgebaut, dass sie auch dann noch sinnvoll bleibt, wenn später weitere Notebooks hinzukommen.

---

# Projektüberblick

In diesem Projekt wird eine Machine-Learning-Pipeline aufgebaut, die für Kreditdaten vorhersagen soll, ob ein Zahlungsausfall wahrscheinlich ist.

Die Zielvariable ist:

- `not.fully.paid`
- `0` = Kredit wurde vollständig zurückgezahlt
- `1` = Kredit wurde nicht vollständig zurückgezahlt

Der bisherige Workflow besteht aus mehreren aufeinander aufbauenden Schritten:

1. Daten laden und verstehen
2. Daten prüfen und validieren
3. Daten vorbereiten und Features erzeugen
4. Modelle trainieren und vergleichen
5. Modellqualität bewerten
6. Modell über eine API nutzbar machen

Weitere Schritte können später ergänzt werden, zum Beispiel Monitoring, Tests, Deployment oder Automatisierung.

---

# Grundidee des Projekts

Man kann sich das Projekt wie eine Verarbeitungskette vorstellen:

- Zuerst werden die Rohdaten geprüft.
- Danach werden sie für das Machine Learning vorbereitet.
- Anschließend werden mehrere Modelle trainiert.
- Das beste Modell wird bewertet.
- Zum Schluss wird es so bereitgestellt, dass neue Eingaben getestet werden können.

Die Notebooks bauen logisch aufeinander auf.  
Deshalb sollte man sie möglichst in der vorgesehenen Reihenfolge lesen und ausführen.

---

# Arbeitsweise mit den Notebooks

Die Notebooks sind nicht als isolierte Einzeldateien gedacht.  
Jedes Notebook übernimmt eine bestimmte Aufgabe im Gesamtprozess und erzeugt Ergebnisse, die später wiederverwendet werden.

Typischer Ablauf:

- Ein Notebook liest Daten oder Modelle aus einem vorherigen Schritt ein.
- Es verarbeitet diese weiter.
- Anschließend speichert es ein Ergebnis, das in einem späteren Notebook verwendet wird.

Darum ist es wichtig, beim Lesen immer auf diese drei Fragen zu achten:

1. **Was ist die Eingabe?**
2. **Was passiert im Notebook?**
3. **Was ist die Ausgabe?**

---

# Wichtige Python-Grundbegriffe

## DataFrame

Ein `DataFrame` ist eine Tabelle in Python.  
Er stammt meistens aus der Bibliothek `pandas`.

Man kann ihn sich wie eine Excel-Tabelle vorstellen:

- jede Zeile = ein Datensatz
- jede Spalte = ein Merkmal

In diesem Projekt bedeutet das zum Beispiel:

- eine Zeile = ein Kreditfall
- eine Spalte = z. B. `fico`, `dti` oder `int.rate`

---

## Variable

Eine Variable speichert einen Wert.

Beispiel:

```python
x = 5
```

Hier speichert `x` den Wert `5`.

---

## Funktion

Eine Funktion ist ein benannter Codeblock, der eine bestimmte Aufgabe erfüllt.

Beispiel:

```python
def addiere(a, b):
    return a + b
```

Die Funktion bekommt zwei Werte und gibt ein Ergebnis zurück.

Wichtige Bestandteile:

- `def` startet die Funktion
- `a` und `b` sind Eingaben
- `return` gibt das Ergebnis zurück

---

## Klasse

Eine Klasse ist ein Bauplan für zusammengehörige Funktionen und Daten.

Im Projekt wird z. B. eine Klasse für die Vorverarbeitung verwendet.  
Das ist sinnvoll, wenn mehrere Arbeitsschritte logisch zusammengehören.

---

## Methode

Eine Methode ist eine Funktion innerhalb einer Klasse.

Beispiele aus dem Projekt:

- `clean_data()`
- `engineer_features()`
- `transform()`

---

## Import

Mit `import` werden Bibliotheken geladen.

Beispiel:

```python
import pandas as pd
```

Das bedeutet:

- die Bibliothek `pandas` wird geladen
- `pd` ist nur eine Kurzform

---

## Dictionary

Ein Dictionary speichert Werte in Form von Schlüssel-Wert-Paaren.

Beispiel:

```python
person = {"name": "Anna", "alter": 25}
```

Hier sind:

- `"name"` und `"alter"` die Schlüssel
- `"Anna"` und `25` die Werte

---

# Aufbau der bisherigen Notebooks

Die folgenden Abschnitte beschreiben den bisherigen Stand des Projekts.  
Wenn später weitere Notebooks hinzukommen, kann diese Dokumentation einfach erweitert werden.

---

# Notebook 01 – Datenexploration und Validierung

## Ziel

Dieses Notebook dient als Einstieg in die Daten.

Hier werden die Rohdaten:

- geladen
- untersucht
- auf Auffälligkeiten geprüft
- validiert
- als bereinigte Zwischenstufe gespeichert

Man kann dieses Notebook als Eingangskontrolle des Projekts sehen.

---

## Typische Eingabe

```text
data/raw/loan_data.csv
```

---

## Typische Ausgabe

```text
data/processed/loan_data_validated.csv
```

---

## Was inhaltlich passiert

### Daten laden

Typischer Code:

```python
df = pd.read_csv("../data/raw/loan_data.csv")
```

### Erklärung

- `pd.read_csv(...)` liest eine CSV-Datei ein
- das Ergebnis wird in `df` gespeichert
- `df` ist danach eine Tabelle, mit der weitergearbeitet wird

---

### Erste Sicht auf die Daten

Typische Befehle:

```python
df.head()
df.info()
df.describe()
```

#### `df.head()`
Zeigt die ersten Zeilen der Tabelle.  
Das hilft, um die Struktur und erste Beispielwerte zu sehen.

#### `df.info()`
Zeigt:

- wie viele Zeilen vorhanden sind
- welche Spalten existieren
- welche Datentypen verwendet werden
- ob fehlende Werte vorliegen

#### `df.describe()`
Zeigt statistische Kennzahlen numerischer Spalten, z. B.:

- Mittelwert
- Minimum
- Maximum
- Standardabweichung
- Quartile

---

### Fehlende Werte prüfen

Typischer Code:

```python
df.isnull().sum()
```

### Erklärung

- `isnull()` prüft, ob Werte fehlen
- `sum()` zählt diese Fälle

So erkennt man schnell, ob einzelne Spalten Lücken enthalten.

---

### Doppelte Zeilen prüfen

Typischer Code:

```python
df.duplicated().sum()
```

### Erklärung

Hier wird geprüft, ob identische Datensätze mehrfach vorkommen.

Doppelte Zeilen können problematisch sein, weil sie das Modell verzerren.

---

### Datenvalidierung

In diesem Schritt werden Regeln für die Datenqualität definiert.

Beispiele:

- eine Spalte darf nicht leer sein
- ein Wert muss innerhalb eines sinnvollen Bereichs liegen
- eine Spalte muss den richtigen Datentyp haben

Wenn Great Expectations verwendet wird, können diese Regeln automatisch geprüft werden.

### Einfache Vorstellung

Datenvalidierung ist eine Art Qualitätskontrolle für den Datensatz.

---

### Validierte Daten speichern

Typischer Code:

```python
df.to_csv("../data/processed/loan_data_validated.csv", index=False)
```

### Erklärung

- `to_csv(...)` speichert den DataFrame als Datei
- `index=False` verhindert eine zusätzliche unnötige Index-Spalte

---

## Warum dieses Notebook wichtig ist

Wenn Fehler in den Rohdaten hier nicht erkannt werden, setzen sie sich in allen späteren Schritten fort.

Typische Folgen wären:

- fehlerhafte Vorverarbeitung
- ungenaue Features
- schlechte Modellleistung
- unzuverlässige Vorhersagen

---

# Notebook 02 – Datenvorverarbeitung

## Ziel

In diesem Notebook werden die validierten Daten so vorbereitet, dass sie für das Modelltraining verwendet werden können.

Dazu gehören vor allem:

- Datenbereinigung
- Feature Engineering
- Umwandlung kategorialer Spalten
- Speichern des vorverarbeiteten Datensatzes

---

## Typische Eingabe

```text
data/processed/loan_data_validated.csv
```

---

## Typische Ausgabe

```text
data/processed/loan_data_preprocessed.csv
```

---

## Warum Vorverarbeitung nötig ist

Rohdaten sind meist noch nicht in der Form, die ein Modell direkt nutzen kann.

Probleme können zum Beispiel sein:

- fehlende Werte
- doppelte Zeilen
- Textspalten
- Merkmale, die noch nicht aussagekräftig genug sind

Die Vorverarbeitung macht aus Rohdaten einen Datensatz, mit dem ein Modell sinnvoll arbeiten kann.

---

## Die Klasse `PreprocessingPipeline`

In diesem Notebook gibt es typischerweise eine Klasse wie:

```python
class PreprocessingPipeline:
    ...
```

### Warum ist das sinnvoll?

Eine Klasse bündelt mehrere zusammengehörige Schritte.

Das macht den Ablauf übersichtlicher, weil man sofort erkennt:

- welche Schritte es gibt
- in welcher Reihenfolge sie ausgeführt werden
- welche Methode wofür zuständig ist

---

## Methode `clean_data()`

Diese Methode ist für die Bereinigung zuständig.

Typischer Code:

```python
df = df.dropna()
df = df.drop_duplicates()
```

### Erklärung

#### `dropna()`
Entfernt Zeilen mit fehlenden Werten.

#### `drop_duplicates()`
Entfernt doppelte Zeilen.

Diese Schritte sorgen dafür, dass der Datensatz konsistenter wird.

---

## Methode `engineer_features()`

Hier werden zusätzliche Merkmale erstellt, die dem Modell helfen sollen.

Typische Features sind:

- `debt_burden_ratio`
- `dti_to_fico`
- `int_rate_to_fico`
- `revol_bal_to_inc`

---

## Warum Feature Engineering sinnvoll ist

Nicht immer reichen die ursprünglichen Spalten aus, um gute Vorhersagen zu ermöglichen.

Manchmal sind Kombinationen aus bestehenden Merkmalen informativer als die Einzelwerte.

Beispiele:

- hohe Schuldenquote + schlechter FICO-Score
- hoher Revolving Balance im Verhältnis zum Einkommen
- Zinssatz im Verhältnis zur Kreditwürdigkeit

---

## Beispiel: `debt_burden_ratio`

Typischer Code:

```python
df["debt_burden_ratio"] = df["installment"] * df["dti"] / np.exp(df["log.annual.inc"])
```

### Erklärung

Dieses Feature versucht, die finanzielle Belastung eines Kreditnehmers abzubilden.

Bestandteile:

- `installment` = monatliche Rate
- `dti` = Schuldenquote
- `log.annual.inc` = logarithmiertes Einkommen

### Warum `np.exp(...)`?

Das Einkommen liegt im Datensatz logarithmiert vor.  
Mit `np.exp(...)` wird dieser Logarithmus wieder rückgängig gemacht, damit der Wert besser weiterverwendet werden kann.

---

## Beispiel: `dti_to_fico`

Typischer Code:

```python
df["dti_to_fico"] = df["dti"] / df["fico"]
```

### Erklärung

Hier werden Schuldenquote und Kredit-Score miteinander kombiniert.

Ein hoher Wert kann ein Hinweis auf ein höheres Risiko sein.

---

## Beispiel: `int_rate_to_fico`

Typischer Code:

```python
df["int_rate_to_fico"] = df["int.rate"] / df["fico"]
```

### Erklärung

Dieses Feature verbindet Zinssatz und Kreditwürdigkeit in einer Kennzahl.

---

## Beispiel: `revol_bal_to_inc`

Typischer Code:

```python
df["revol_bal_to_inc"] = df["revol.bal"] / np.exp(df["log.annual.inc"])
```

### Erklärung

Hier wird der Revolving Balance ins Verhältnis zum Einkommen gesetzt.

Auch das kann ein Hinweis auf finanzielle Belastung sein.

---

## Methode `encode_categorical()`

Modelle können mit Textwerten meist nicht direkt arbeiten.

Beispiel:

- `"credit_card"`
- `"small_business"`
- `"debt_consolidation"`

Darum werden kategoriale Variablen in numerische Form umgewandelt.

Typischer Code:

```python
df = pd.get_dummies(df, columns=["purpose"], drop_first=True, dtype=int)
```

### Erklärung

Aus einer Textspalte entstehen mehrere 0/1-Spalten, z. B.:

- `purpose_credit_card`
- `purpose_small_business`

Bedeutung:

- `1` = Kategorie trifft zu
- `0` = Kategorie trifft nicht zu

---

## Warum `drop_first=True`?

Dadurch wird eine Kategorie weggelassen, um Redundanz zu vermeiden.

Das ist besonders bei linearen Modellen sinnvoll.

---

## Methode `save_data()`

Diese Methode speichert den verarbeiteten Datensatz.

Typischer Code:

```python
df.to_csv("../data/processed/loan_data_preprocessed.csv", index=False)
```

---

## Methode `transform()`

Diese Methode ruft die einzelnen Schritte in der richtigen Reihenfolge auf.

Beispiel:

```python
df = self.clean_data(df)
df = self.engineer_features(df)
df = self.encode_categorical(df)
df = self.save_data(df, filepath)
```

### Erklärung

`transform()` ist die zentrale Methode der Vorverarbeitung.  
Sie startet den gesamten Ablauf.

---

## Warum dieses Notebook wichtig ist

Dieses Notebook stellt sicher, dass die Daten in einer Form vorliegen, mit der das Modell später wirklich arbeiten kann.

Ohne diese Schritte wären Training und Vorhersage deutlich fehleranfälliger.

---

# Notebook 03 – Modelltraining und Experiment-Tracking

## Ziel

In diesem Notebook werden mehrere Modelle trainiert, verglichen und in MLflow dokumentiert.

Am Ende wird das beste Modell registriert.

---

## Typische Eingabe

```text
data/processed/loan_data_preprocessed.csv
```

---

## Typische Ausgaben

Dieses Notebook erzeugt in der Regel:

- Trainingsläufe in MLflow
- geloggte Parameter
- geloggte Metriken
- ein registriertes Modell

---

## Aufteilung in Eingaben und Ziel

Typischer Code:

```python
X = processed_data.drop(columns="not.fully.paid")
y = processed_data["not.fully.paid"]
```

### Erklärung

- `X` enthält die Eingabemerkmale
- `y` enthält die Zielvariable

Das Modell soll aus `X` lernen, `y` vorherzusagen.

---

## Train-Test-Split

Typischer Code:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Erklärung

Die Daten werden in zwei Teile geteilt:

- Trainingsdaten zum Lernen
- Testdaten zur späteren Überprüfung

### Wichtige Parameter

#### `test_size=0.2`
20 % der Daten werden für den Test reserviert.

#### `random_state=42`
Die Aufteilung bleibt reproduzierbar.

#### `stratify=y`
Die Verteilung der Zielvariable bleibt in Training und Test ähnlich.

Das ist besonders bei ungleich verteilten Klassen wichtig.

---

## Welche Modelle werden trainiert?

Typischerweise werden mehrere Modelle ausprobiert, z. B.:

- Logistic Regression
- Random Forest
- Gradient Boosting

Warum mehrere Modelle?

Weil nicht vorher sicher ist, welches Modell für den Datensatz am besten geeignet ist.

---

## Logistic Regression

Die Logistic Regression ist ein klassisches Modell für binäre Klassifikation.

Oft wird sie als Pipeline mit Skalierung aufgebaut:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(...))
])
```

### Warum eine Pipeline?

Die Pipeline verbindet mehrere Schritte sauber miteinander:

1. Daten skalieren
2. Modell trainieren

Das verhindert Fehler und macht den Ablauf nachvollziehbarer.

---

## StandardScaler

`StandardScaler()` bringt numerische Variablen auf eine ähnliche Skala.

Das ist besonders für Modelle wie Logistic Regression sinnvoll.

Wichtig:
Baumbasierte Modelle wie Random Forest oder Gradient Boosting brauchen diese Skalierung meistens nicht.

---

## Random Forest

Random Forest ist ein baumbasiertes Verfahren.

Typische Eigenschaften:

- robust
- kann nichtlineare Zusammenhänge lernen
- oft gute Grundperformance
- braucht normalerweise keine Skalierung

---

## Gradient Boosting

Gradient Boosting ist ebenfalls ein baumbasiertes Verfahren.

Hier werden mehrere Modelle nacheinander aufgebaut.  
Spätere Modelle versuchen, Fehler früherer Modelle zu korrigieren.

---

## Funktion `train_and_evaluate_model(...)`

Diese Funktion übernimmt typischerweise den kompletten Ablauf für einen Modelllauf:

1. Modell trainieren
2. Vorhersagen berechnen
3. Metriken berechnen
4. Ergebnisse in MLflow speichern

Das ist praktisch, weil derselbe Ablauf für verschiedene Modelle wiederverwendet werden kann.

---

## Wichtige Metriken

### Accuracy
Wie viele Vorhersagen insgesamt richtig waren.

### Precision
Wenn das Modell „Ausfall“ sagt: Wie oft ist das korrekt?

### Recall
Wie viele tatsächliche Ausfälle wurden erkannt?

### F1-Score
Kombination aus Precision und Recall.

### ROC-AUC
Misst, wie gut das Modell zwischen den Klassen trennt.

---

## Warum mehrere Metriken nötig sind

Bei unausgeglichenen Klassen reicht Accuracy allein oft nicht aus.

Ein Modell kann eine gute Accuracy haben und trotzdem Ausfälle schlecht erkennen.  
Deshalb sind Precision, Recall, F1 und ROC-AUC wichtig.

---

## Was ist MLflow?

MLflow dokumentiert Machine-Learning-Experimente.

Es speichert zum Beispiel:

- Modelltyp
- Parameter
- Metriken
- Modellartefakte
- Versionen

### Warum ist das wichtig?

Dadurch bleibt nachvollziehbar:

- welches Modell trainiert wurde
- mit welchen Einstellungen
- mit welchem Ergebnis

Das ist für MLOps und Teamarbeit zentral.

---

## Auswahl des besten Modells

Nach mehreren Trainingsläufen wird ein Modell als bestes Modell ausgewählt.

Wichtig ist, dass diese Auswahl wirklich auf Basis aller getesteten Modelle erfolgt.

---

## Modellregistrierung

Das beste Modell wird registriert und bekommt dadurch eine Version.

Beispiel:

- Version 1
- Version 2
- Version 3

So bleibt nachvollziehbar, welches Modell aktuell gültig ist.

---

## Staging-Status

Ein Modell im Status `Staging` ist der aktuelle Kandidat für Tests oder Nutzung.

---

## Warum dieses Notebook wichtig ist

Hier entscheidet sich, welches Modell später bewertet und über die API verwendet wird.

Dieses Notebook ist damit der zentrale Trainingsschritt des Projekts.

---

# Notebook 04 – Modellbewertung

## Ziel

In diesem Notebook wird das beste Modell genauer analysiert.

Während im vorherigen Notebook vor allem trainiert und verglichen wurde, geht es hier um die Frage, wie gut das beste Modell tatsächlich arbeitet.

---

## Typische Eingaben

- `loan_data_preprocessed.csv`
- registriertes Modell aus MLflow

---

## Was hier geprüft wird

Typische Fragen sind:

- Wie stabil ist das Modell?
- Wie gut erkennt es Zahlungsausfälle?
- Welche Fehler macht es?
- Wie verlässlich sind die Ergebnisse?

---

## Modell aus MLflow laden

Das Modell wird nicht neu trainiert, sondern aus MLflow geladen.

Das ist wichtig, weil genau das Modell bewertet werden soll, das später auch verwendet wird.

---

## Testdaten vorbereiten

Wie schon zuvor werden die Daten wieder in `X` und `y` getrennt.

Dabei ist wichtig, dass die Vorbereitung der Daten zur Trainingslogik passt.

Wenn im Modell bereits eine Pipeline mit Skalierung steckt, dürfen die Daten hier nicht noch einmal falsch zusätzlich transformiert werden.

---

## Typische Auswertungen

In diesem Notebook werden oft folgende Kennzahlen und Visualisierungen verwendet:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC-Kurve
- Precision-Recall-Kurve

---

## Confusion Matrix

Die Confusion Matrix zeigt, wie sich die Vorhersagen aufteilen.

Sie hilft bei Fragen wie:

- Wie viele Ausfälle wurden richtig erkannt?
- Wie viele Nicht-Ausfälle wurden richtig erkannt?
- Welche Fehlerarten treten auf?

Gerade für den Einstieg ist das sehr hilfreich, weil man das Verhalten des Modells dadurch besser versteht.

---

## Cross-Validation

Cross-Validation bedeutet, dass das Modell mehrfach mit unterschiedlichen Datenaufteilungen geprüft wird.

Dadurch erhält man eine stabilere Einschätzung als mit nur einem einzigen Testsplit.

---

## Beispielhafte Funktion `validate_model_cv(...)`

Diese Funktion berechnet typischerweise Mittelwerte und Streuungen über mehrere Folds.

### Erklärung

Das Modell wird mehrfach geprüft, nicht nur einmal.  
Danach wird zusammengefasst, wie stabil die Leistung insgesamt ist.

---

## Warum dieses Notebook wichtig ist

Hier zeigt sich, ob das ausgewählte Modell fachlich und technisch überzeugt.

Das ist wichtig für:

- die Bewertung der Modellqualität
- die spätere Nutzung
- die Dokumentation der Ergebnisse
- die Nachvollziehbarkeit des Projekts

---

# Notebook 05 – API und Testvorhersagen

## Ziel

In diesem Notebook wird das Modell über eine API ansprechbar gemacht.

Das bedeutet:
Neue Daten können an die API gesendet werden, und die API gibt eine Vorhersage zurück.

---

## Warum eine API sinnvoll ist

Ohne API müsste jede Vorhersage direkt im Python-Code ausgeführt werden.

Mit API ist das Modell über klar definierte Endpunkte erreichbar und kann später leichter in andere Anwendungen eingebunden werden.

---

## Wichtige Bibliotheken

### FastAPI
Wird verwendet, um die API und ihre Endpunkte zu definieren.

### Pydantic
Prüft, ob Eingaben vollständig und korrekt formatiert sind.

### TestClient
Erlaubt es, die API direkt im Notebook zu testen, ohne einen externen Server zu starten.

---

## Modell laden

Auch hier wird das registrierte Modell aus MLflow geladen.

Damit greift die API auf genau das Modell zu, das vorher ausgewählt wurde.

---

## Eingabemodell mit Pydantic

Typischer Aufbau:

```python
class LoanFeatures(BaseModel):
    ...
```

### Warum ist das hilfreich?

Damit wird festgelegt:

- welche Felder eingegeben werden müssen
- welche Datentypen erlaubt sind
- welche Wertebereiche gültig sind

So erkennt die API fehlerhafte Eingaben frühzeitig.

---

## Funktion `preprocess_input(...)`

Diese Funktion ist besonders wichtig.

Sie sorgt dafür, dass ein neuer API-Input in genau die Form gebracht wird, die das Modell erwartet.

Typische Aufgaben:

1. Eingabewerte übernehmen
2. neue Features berechnen
3. Dummy-Spalten erzeugen
4. Spalten in die richtige Reihenfolge bringen
5. Ergebnis als DataFrame zurückgeben

---

## Warum ist das so wichtig?

Das Modell wurde mit einer bestimmten Datenstruktur trainiert.

Wenn die API eine andere Struktur liefert, entstehen schnell:

- Fehler
- inkonsistente Ergebnisse
- unplausible Vorhersagen

Deshalb muss die Vorverarbeitung in der API mit der Trainingslogik zusammenpassen.

---

## FastAPI-App

Typischer Start:

```python
app = FastAPI(...)
```

### Hinweis

Die App sollte nur einmal definiert werden.  
Wenn `app` mehrfach neu erstellt wird, können bereits definierte Routen überschrieben werden.

---

## Typische Endpunkte

### `/`
Liefert Basisinformationen zur API.

### `/health`
Zeigt, ob die API läuft und das Modell geladen ist.

### `/predict`
Nimmt einen einzelnen Datensatz entgegen und gibt eine Vorhersage zurück.

### `/predict_batch`
Verarbeitet mehrere Datensätze auf einmal.

---

## Ablauf eines `/predict`-Requests

1. Ein JSON mit Eingabedaten wird gesendet
2. Pydantic prüft die Eingaben
3. `preprocess_input(...)` erzeugt die Modellstruktur
4. `model.predict(...)` berechnet die Klasse
5. optional liefert `predict_proba(...)` eine Wahrscheinlichkeit
6. die API gibt eine JSON-Antwort zurück

---

## Beispiel einer API-Antwort

```json
{
  "prediction": 1,
  "default_risk_probability": 0.73,
  "interpretation": "Zahlungsausfall wahrscheinlich"
}
```

---

## Warum `TestClient` im Notebook sinnvoll ist

Für die Entwicklung ist es oft einfacher, die API direkt im Notebook zu testen.

Dann braucht man nicht zusätzlich:

- eine separate `main.py`
- `uvicorn.run(...)`
- einen laufenden Serverprozess

Beispiel:

```python
test_client = TestClient(app)
response = test_client.post("/predict", json=sample_payload)
```

Das ist besonders praktisch für Debugging und Präsentation.

---

## Typische Fehler in diesem Notebook

### `app` mehrfach definieren
Dann können Routen oder Einstellungen überschrieben werden.

### `test_client` verwenden, bevor er erstellt wurde
Dann kennt Python das Objekt noch nicht.

### Unterschiedliche Vorverarbeitung in Training und API
Dann passen die Eingabespalten nicht mehr zum Modell.

### Unrealistische Testdaten
Dann können Vorhersagen unlogisch wirken, obwohl die API technisch korrekt funktioniert.

---

# Zusammenhang der Notebooks

Die bisherigen Notebooks bilden zusammen eine durchgehende Pipeline:

1. Rohdaten prüfen
2. Daten vorbereiten
3. Modelle trainieren
4. Modell bewerten
5. Vorhersagen über API ermöglichen

Wenn später weitere Notebooks ergänzt werden, sollten sie an dieser Kette anknüpfen und klar beschreiben:

- welche Eingabe sie verwenden
- was sie verarbeiten
- welche Ausgabe sie erzeugen

---

# Typischer Datenfluss im Projekt

```text
data/raw/loan_data.csv
    ↓
data/processed/loan_data_validated.csv
    ↓
data/processed/loan_data_preprocessed.csv
    ↓
MLflow Runs und Modellregistry
    ↓
API für neue Vorhersagen


```

---

# Häufige Fehlerquellen

## Falsche Reihenfolge der Notebooks
Wenn frühere Dateien noch nicht erzeugt wurden, schlagen spätere Schritte fehl.

## Unterschiedliche Spaltenstrukturen
Wenn Training, Evaluation und API nicht dieselbe Struktur verwenden, entstehen Fehler oder inkonsistente Ergebnisse.

## Modell nicht in MLflow registriert
Dann kann es in späteren Schritten nicht geladen werden.

## Vorverarbeitung nicht konsistent
Das Modell muss dieselbe Logik in Training, Bewertung und API sehen.

## Unrealistische Testdaten
Machine-Learning-Modelle verhalten sich außerhalb des gelernten Wertebereichs oft unvorhersehbar.

---

# Wie man sich am besten in das Projekt einarbeitet

Ein sinnvoller Einstieg ist:

1. zuerst den Datenfluss verstehen
2. dann die Reihenfolge der Notebooks ansehen
3. danach pro Notebook Eingabe, Verarbeitung und Ausgabe notieren
4. erst im letzten Schritt tiefer in einzelne Funktionen und Codezeilen gehen

Hilfreiche Fragen beim Lesen des Codes:

- Welche Datei wird eingelesen?
- Was wird im Notebook verändert?
- Welche Datei oder welches Modell kommt am Ende heraus?
- Welche Funktion ist für welchen Schritt zuständig?
- Welche Teile müssen später wiederverwendet werden?

---

# Fazit

Das Projekt bildet eine vollständige Machine-Learning-Pipeline für Kreditdaten ab:

- Daten laden und prüfen
- Daten vorbereiten
- Modelle trainieren
- das beste Modell bewerten
- Vorhersagen über eine API ermöglichen

Der wichtigste Punkt beim Einstieg ist nicht, sofort jede einzelne Python-Zeile perfekt zu verstehen.  
Entscheidend ist zuerst der Gesamtzusammenhang: Datenfluss, Reihenfolge der Schritte und Aufgabe jedes Notebooks.

Wenn dieser Ablauf klar ist, wird auch der Code deutlich verständlicher.

---

# Kurzfassung in einem Satz

Dieses Projekt lädt Kreditdaten, bereitet sie für Machine Learning auf, trainiert und bewertet Modelle und stellt das ausgewählte Modell anschließend für neue Vorhersagen bereit.
