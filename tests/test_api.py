from fastapi.testclient import TestClient

from Code.main import app

client = TestClient(app)

FLUG = {
    "airline": "Vistara", "von": "Delhi", "nach": "Mumbai",
    "abflugzeit": "Morning", "stopps": "zero", "klasse": "Economy",
    "tage_bis_abflug": 30,
}


def test_status():
    assert client.get("/").json()["status"] == "ok"


def test_modell_info():
    info = client.get("/modell-info").json()
    assert info["version"]
    assert info["r2"] > 0.9


def test_flugpreis_check_liefert_preis_und_empfehlung():
    r = client.post("/flugpreis-check", json=FLUG)
    assert r.status_code == 200
    d = r.json()
    assert d["preis_heute_chf"] > 0
    assert d["empfehlung"].startswith(("Jetzt buchen", "Warten"))
    assert len(d["preisverlauf"]) == 30
    assert d["bester_preis_chf"] <= d["preis_heute_chf"]


def test_business_teurer_als_economy():
    eco = client.post("/flugpreis-check", json=FLUG).json()["preis_heute_chf"]
    bus = client.post("/flugpreis-check", json={**FLUG, "klasse": "Business"}).json()["preis_heute_chf"]
    assert bus > eco


def test_gleiche_stadt_abgelehnt():
    assert client.post("/flugpreis-check", json={**FLUG, "nach": "Delhi"}).status_code == 422


def test_ungueltige_tage_abgelehnt():
    assert client.post("/flugpreis-check", json={**FLUG, "tage_bis_abflug": 0}).status_code == 422
    assert client.post("/flugpreis-check", json={**FLUG, "tage_bis_abflug": 50}).status_code == 422
