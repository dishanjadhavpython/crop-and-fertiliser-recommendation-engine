"""The console is served from the same origin as the API.

Thin on purpose — the visual layer is checked by eye, but the wiring that
makes it reachable, and the response fields it depends on, are worth guarding.
"""
import warnings

import pytest
from fastapi.testclient import TestClient

from src.serve.api import app

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_console_is_served_at_the_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Regur" in r.text
    assert "/static/app.css" in r.text and "/static/app.js" in r.text


@pytest.mark.parametrize("asset", ["app.css", "app.js"])
def test_static_assets_are_reachable(client, asset):
    assert client.get(f"/static/{asset}").status_code == 200


def test_recommend_returns_every_field_the_console_renders(client):
    r = client.post("/recommend", json={
        "district": "KOLHAPUR", "taluka": "KARVIR", "season": "Kharif", "top_k": 5})
    assert r.status_code == 200
    d = r.json()
    for key in ("district", "season", "soil_class", "confident", "water_limited",
                "crops", "vetoed", "micronutrients", "context"):
        assert key in d, key

    crop = d["crops"][0]
    for key in ("crop", "crop_marathi", "rank", "suitability_class",
                "yield_class", "yield_regime", "reason", "factors",
                "requires_irrigation", "fertiliser"):
        assert key in crop, key

    # the Liebig widget needs all eight factor scores, and the gate's score
    # must be their minimum or the chart contradicts the number beside it
    assert len(crop["factors"]) == 8
    assert min(crop["factors"].values()) == pytest.approx(crop["rule_score"], abs=1e-3)


def test_context_carries_what_the_strip_displays(client):
    ctx = client.post("/recommend", json={
        "district": "SOLAPUR", "taluka": "SANGOLE", "season": "Rabi"}).json()["context"]
    for key in ("annual_rainfall_mm", "rootzone_awc_mm", "aridity_index",
                "lgp_days", "shc_samples"):
        assert key in ctx and ctx[key] is not None


def test_recommend_accepts_all_twelve_soil_health_card_components(client):
    """The card prints twelve values; the request body must carry all twelve."""
    r = client.post("/recommend", json={
        "district": "SOLAPUR", "taluka": "SANGOLE", "season": "Kharif",
        "soil_test": {
            "n_kg_ha": 400, "p_kg_ha": 17, "k_kg_ha": 190, "oc_pct": 0.6,
            "ph": 6.8, "ec_status": "normal",
            "sulphur_status": "low", "zinc_status": "low", "iron_status": "normal",
            "copper_status": "normal", "boron_status": "normal", "manganese_status": "normal",
        },
    })
    assert r.status_code == 200
    d = r.json()
    assert d["context"]["ph_source"] == "farmer soil health card"
    assert d["context"]["ph_used"] == pytest.approx(6.8)
    assert d["context"]["ec_source"] == "farmer soil health card"
    by_component = {m["component"]: m for m in d["micronutrients"]}
    assert by_component["S"]["source"] == "farmer soil health card"
    assert by_component["Zn"]["source"] == "farmer soil health card"


def test_unknown_taluka_is_a_404_not_a_crash(client):
    r = client.get("/recommend", params={"district": "X", "taluka": "Y", "season": "Rabi"})
    assert r.status_code == 404


def test_bad_season_is_rejected(client):
    r = client.get("/recommend", params={
        "district": "KOLHAPUR", "taluka": "KARVIR", "season": "Monsoon"})
    assert r.status_code == 400
