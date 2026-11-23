"""Trainiert das Flugpreis-Modell und speichert es unter model/flugpreis_model.joblib.

Datensatz: "Flight Price Prediction" (Kaggle, Shubham Bathwal), ca. 300'000 Flüge
zwischen den sechs grössten Städten Indiens. Preis in indischen Rupien (INR).

Aufruf (im Projektordner):  python Code/train.py
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Clean_Dataset.csv"
MODEL = ROOT / "model" / "flugpreis_model.joblib"
MODELL_VERSION = "1.0.0"   # bei jedem Neutraining erhöhen

KATEGORIEN = ["airline", "source_city", "destination_city", "departure_time", "stops", "class"]
NUMERISCH = ["days_left"]
FEATURES = KATEGORIEN + NUMERISCH


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vorverarbeitung = ColumnTransformer(
        [("kat", OrdinalEncoder(), KATEGORIEN)], remainder="passthrough"
    )
    modell = HistGradientBoostingRegressor(
        categorical_features=list(range(len(KATEGORIEN))),
        max_iter=500,
        learning_rate=0.1,
        random_state=42,
    )
    pipeline = Pipeline([("vorverarbeitung", vorverarbeitung), ("modell", modell)])

    # Preise sind stark schief verteilt (Economy vs. Business) -> auf log-Skala lernen
    pipeline.fit(X_train, np.log(y_train))
    vorhersage = np.exp(pipeline.predict(X_test))

    r2 = r2_score(y_test, vorhersage)
    mae = mean_absolute_error(y_test, vorhersage)
    print(f"Trainingsdaten: {len(X_train):,} Flüge | Testdaten: {len(X_test):,} Flüge")
    print(f"R²:  {r2:.3f}")
    print(f"MAE: {mae:,.0f} INR")

    info = {
        "version": MODELL_VERSION,
        "trainiert_am": date.today().isoformat(),
        "algorithmus": "HistGradientBoostingRegressor (log-Preis)",
        "scikit_learn": sklearn.__version__,
        "trainingsdaten": len(X_train),
        "testdaten": len(X_test),
        "r2": round(r2, 3),
        "mae_inr": round(mae),
    }
    werte = {spalte: sorted(df[spalte].unique().tolist()) for spalte in KATEGORIEN}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": FEATURES, "werte": werte, "info": info}, MODEL)
    print(f"Modell gespeichert: {MODEL.relative_to(ROOT)} ({MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
