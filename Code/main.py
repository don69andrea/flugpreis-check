"""Flugpreis-Check API

Sagt den Preis eines Inlandflugs in Indien voraus und empfiehlt, ob man jetzt buchen
oder besser warten soll. Das ML-Modell wird mit Code/train.py trainiert.

Lokal starten (im Projektordner):  uvicorn Code.main:app --reload
"""
import logging
import time
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PFAD = Path(__file__).resolve().parent.parent / "model" / "flugpreis_model.joblib"
INR_ZU_CHF = 0.0095          # ungefährer Wechselkurs, nur zur Orientierung
MAX_TAGE = 49                # Das Modell kennt nur Buchungen 1-49 Tage vor Abflug
SCHWELLE_WARTEN = 0.05       # Warten lohnt sich erst ab 5 % Ersparnis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("flugpreis-check")

_modell = joblib.load(MODEL_PFAD)
pipeline = _modell["pipeline"]
FEATURES = _modell["features"]
MODELL_INFO = _modell["info"]
log.info("Modell geladen: Version %s (R² %s)", MODELL_INFO["version"], MODELL_INFO["r2"])


# Auswahllisten -> erscheinen in /docs als Dropdown
class Airline(str, Enum):
    AirAsia = "AirAsia"
    Air_India = "Air_India"
    GO_FIRST = "GO_FIRST"
    Indigo = "Indigo"
    SpiceJet = "SpiceJet"
    Vistara = "Vistara"


class Stadt(str, Enum):
    Bangalore = "Bangalore"
    Chennai = "Chennai"
    Delhi = "Delhi"
    Hyderabad = "Hyderabad"
    Kolkata = "Kolkata"
    Mumbai = "Mumbai"


class Abflugzeit(str, Enum):
    Early_Morning = "Early_Morning"
    Morning = "Morning"
    Afternoon = "Afternoon"
    Evening = "Evening"
    Night = "Night"
    Late_Night = "Late_Night"


class Stopps(str, Enum):
    zero = "zero"
    one = "one"
    two_or_more = "two_or_more"


class Klasse(str, Enum):
    Economy = "Economy"
    Business = "Business"


# Eingabe-Schema
class FlugRequest(BaseModel):
    airline: Airline
    von: Stadt
    nach: Stadt
    abflugzeit: Abflugzeit
    stopps: Stopps
    klasse: Klasse
    tage_bis_abflug: int = Field(ge=1, le=MAX_TAGE, description="Wie viele Tage bis zum Abflug (1-49)")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "airline": "Vistara", "von": "Delhi", "nach": "Mumbai",
                "abflugzeit": "Morning", "stopps": "zero", "klasse": "Economy",
                "tage_bis_abflug": 30,
            }]
        }
    }


# Ausgabe-Schema
class Preispunkt(BaseModel):
    tage_bis_abflug: int
    preis_chf: float


class FlugpreisResponse(BaseModel):
    preis_heute_inr: int
    preis_heute_chf: float
    empfehlung: str
    bester_zeitpunkt_tage_vor_abflug: int
    bester_preis_chf: float
    ersparnis_chf: float
    preisverlauf: list[Preispunkt]


def preise_vorhersagen(req: FlugRequest, tage: list[int]) -> np.ndarray:
    """Preis (INR) für denselben Flug bei verschiedenen Buchungszeitpunkten."""
    zeilen = pd.DataFrame({
        "airline": req.airline.value,
        "source_city": req.von.value,
        "destination_city": req.nach.value,
        "departure_time": req.abflugzeit.value,
        "stops": req.stopps.value,
        "class": req.klasse.value,
        "days_left": tage,
    })[FEATURES]
    return np.exp(pipeline.predict(zeilen))


def chf(inr: float) -> float:
    return round(float(inr) * INR_ZU_CHF, 2)


app = FastAPI(
    title="Flugpreis-Check API",
    description="Jetzt buchen oder warten? KI-Vorhersage für Flugpreise (Datenbasis: 300'000 Flüge in Indien).",
)


@app.middleware("http")
async def anfragen_protokollieren(request: Request, call_next):
    """Protokolliert jede Anfrage mit Statuscode und Antwortzeit (sichtbar in den render.com-Logs)."""
    start = time.perf_counter()
    response = await call_next(request)
    dauer_ms = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, response.status_code, dauer_ms)
    return response


@app.get("/")
def status():
    """Health-Check: Läuft der Service und ist das Modell geladen?"""
    return {"status": "ok", "service": "Flugpreis-Check API", "modell_version": MODELL_INFO["version"], "doku": "/docs"}


@app.get("/modell-info")
def modell_info():
    """Welches Modell ist im Einsatz? Version, Trainingsdatum und Genauigkeit."""
    return {**MODELL_INFO, "genauigkeit_mae_chf": chf(MODELL_INFO["mae_inr"]), "datensatz": "Flight Price Prediction (Kaggle), Inlandflüge Indien"}


@app.post("/flugpreis-check", response_model=FlugpreisResponse)
def flugpreis_check(req: FlugRequest):
    if req.von == req.nach:
        raise HTTPException(status_code=422, detail="Start- und Zielort müssen verschieden sein.")

    # Preis für heute und für jeden späteren Buchungstag bis zum Abflug
    tage = list(range(req.tage_bis_abflug, 0, -1))
    preise = preise_vorhersagen(req, tage)
    preis_heute = preise[0]
    i_best = int(np.argmin(preise))
    bester_preis = preise[i_best]

    if bester_preis < preis_heute * (1 - SCHWELLE_WARTEN):
        empfehlung = f"Warten: Am günstigsten {tage[i_best]} Tage vor Abflug buchen."
    else:
        empfehlung = "Jetzt buchen: Später wird es nicht deutlich günstiger."
        i_best, bester_preis = 0, preis_heute

    log.info(
        "Vorhersage: %s %s->%s %s, %d Tage -> CHF %.2f | %s",
        req.airline.value, req.von.value, req.nach.value, req.klasse.value,
        req.tage_bis_abflug, chf(preis_heute), empfehlung,
    )
    return FlugpreisResponse(
        preis_heute_inr=round(float(preis_heute)),
        preis_heute_chf=chf(preis_heute),
        empfehlung=empfehlung,
        bester_zeitpunkt_tage_vor_abflug=tage[i_best],
        bester_preis_chf=chf(bester_preis),
        ersparnis_chf=chf(preis_heute - bester_preis),
        preisverlauf=[Preispunkt(tage_bis_abflug=t, preis_chf=chf(p)) for t, p in zip(tage, preise)],
    )
