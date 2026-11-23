# Flugpreis-Check API

**Jetzt buchen oder warten?** Eine FastAPI-Anwendung, die mit einem ML-Modell den Preis eines
Fluges vorhersagt und den günstigsten Buchungszeitpunkt empfiehlt.

Selbstständige Arbeit, Modul AI Operations (FHNW, HS26): Aufgabe DevOps lokal & cloud.

## Aufbau

```
Code/train.py          Trainiert das Modell (HistGradientBoosting, R² ≈ 0.96)
Code/main.py           FastAPI-App mit Endpoint POST /flugpreis-check
Code/test_API.ipynb    Notebook zum Aufrufen der API (lokal oder render.com)
model/                 Trainiertes Modell (flugpreis_model.joblib)
tests/                 Automatisierte Tests (pytest)
.github/workflows/     CI: Tests laufen bei jedem Push automatisch
```

## Endpoints

| Methode | Pfad | Zweck |
|---|---|---|
| GET | `/` | Health-Check (Status, Modellversion) |
| GET | `/modell-info` | Modellversion, Trainingsdatum, Genauigkeit (R², MAE) |
| POST | `/flugpreis-check` | Preisvorhersage und Buchungsempfehlung |

Jede Anfrage wird mit Statuscode und Antwortzeit geloggt (lokal im Terminal, in der Cloud in den render.com-Logs).

Ablauf: Client → Uvicorn (ASGI-Server) → FastAPI → ML-Modell → JSON-Antwort.

## Lokal starten

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn Code.main:app --reload
```

Dann <http://127.0.0.1:8000/docs> öffnen → `POST /flugpreis-check` → *Try it out* → *Execute*.

## Beispiel

Anfrage:
```json
{"airline": "Indigo", "von": "Delhi", "nach": "Mumbai", "abflugzeit": "Evening",
 "stopps": "zero", "klasse": "Economy", "tage_bis_abflug": 45}
```
Antwort (gekürzt):
```json
{"preis_heute_chf": 30.04, "empfehlung": "Warten: Am günstigsten 26 Tage vor Abflug buchen.",
 "bester_zeitpunkt_tage_vor_abflug": 26, "bester_preis_chf": 26.35, "ersparnis_chf": 3.69,
 "preisverlauf": [{"tage_bis_abflug": 45, "preis_chf": 30.04}, ...]}
```

## Deployment auf render.com

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn Code.main:app --host 0.0.0.0 --port $PORT`
- Python-Version: über `.python-version` (3.12)
- Auto-Deploy: *After CI Checks Pass* → eine neue Version geht nur online, wenn die Tests grün sind

## Modell neu trainieren

Datensatz *Flight Price Prediction* (Kaggle, Shubham Bathwal) als `data/Clean_Dataset.csv` ablegen,
dann `python Code/train.py`. Der Datensatz ist nicht im Repository enthalten.

## Daten & Grenzen

- ca. 300'000 Inlandflüge zwischen 6 indischen Städten, 6 Airlines, Buchung 1–49 Tage vor Abflug
- Preise in INR, CHF-Umrechnung mit festem Näherungskurs (0.0095)
- Das Modell kennt nur diese Routen und diesen Zeitraum; es ist eine Demo, keine Preisgarantie
