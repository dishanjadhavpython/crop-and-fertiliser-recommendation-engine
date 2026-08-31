"""FastAPI serving layer (plan §9 week 7).

    uvicorn src.serve.api:app --reload

taluka + season + optional Soil Health Card values -> ranked crops with yield
bands, dose plans, and a plain-language reason per recommendation drawn from
the S2 limiting factor.

Marathi crop names come from the fertiliser table's own ``Crop_Local_Name``
column, which already carries them in Devanagari.
"""
from __future__ import annotations

from dataclasses import asdict

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.data.load import taluka_universe
from src.pipeline import load_pipeline, recommend
from src.rules.soil_class import SoilTest

app = FastAPI(
    title="Crop & Fertiliser Recommendation Engine",
    description=__doc__,
    version="1.0",
)


class SoilTestIn(BaseModel):
    """A farmer's own Soil Health Card values. All optional."""
    n_kg_ha: float | None = Field(None, description="Available nitrogen, kg/ha")
    p_kg_ha: float | None = Field(None, description="Available phosphorus, kg/ha")
    k_kg_ha: float | None = Field(None, description="Available potassium, kg/ha")
    oc_pct: float | None = Field(None, description="Organic carbon, %")


class RecommendIn(BaseModel):
    district: str
    taluka: str
    season: str = Field(..., description="Kharif | Rabi | Summer | Whole Year")
    irrigated: bool = False
    top_k: int = 5
    soil_test: SoilTestIn | None = None


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The console. Served from the same origin as the API, so no CORS."""
    return FileResponse(STATIC / "index.html")


@app.on_event("startup")
def _warm() -> None:
    """Fit the learned components once, at startup, not on the first request."""
    load_pipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "talukas": len(taluka_universe())}


@app.get("/talukas")
def talukas(district: str | None = None) -> dict:
    """The 351 Soil Health Card talukas this system can serve."""
    u = taluka_universe()
    if district:
        u = u[u["District"] == district.upper()]
    return {
        "n": len(u),
        "districts": sorted(u["District"].unique()),
        "talukas": u[config.KEY].to_dict("records"),
    }


@app.get("/atlas")
def atlas() -> dict:
    """Every taluka's real coordinates plus the climate it sits in.

    The console draws Maharashtra from these points rather than from a stock
    outline, so the map is the feature store rendered geographically: each dot
    is a taluka at its own latitude and longitude, shaded by aridity.
    """
    import pandas as pd

    from src.features.build_store import build_feature_store

    store = build_feature_store()
    cols = ["District", "Taluka", "Latitude", "Longitude",
            "aridity_annual", "rain_annual", "rootzone_awc", "lgp_annual"]
    df = store[cols].copy()
    return {
        "bounds": {
            "lat": [float(df["Latitude"].min()), float(df["Latitude"].max())],
            "lon": [float(df["Longitude"].min()), float(df["Longitude"].max())],
        },
        "aridity_range": [float(df["aridity_annual"].min()),
                          float(df["aridity_annual"].max())],
        "talukas": [
            {"d": r.District, "t": r.Taluka,
             "y": round(float(r.Latitude), 4), "x": round(float(r.Longitude), 4),
             "a": round(float(r.aridity_annual), 3),
             "r": int(r.rain_annual), "w": int(r.rootzone_awc),
             "l": int(r.lgp_annual)}
            for r in df.itertuples(index=False)
        ],
    }


@app.get("/seasons")
def seasons() -> dict:
    return {"seasons": config.APY_SEASONS}


@app.post("/recommend")
def post_recommend(body: RecommendIn) -> dict:
    return _recommend(body.district, body.taluka, body.season, body.irrigated,
                      body.top_k, body.soil_test)


@app.get("/recommend")
def get_recommend(
    district: str,
    taluka: str,
    season: str = Query(..., description="Kharif | Rabi | Summer | Whole Year"),
    irrigated: bool = False,
    top_k: int = 5,
) -> dict:
    return _recommend(district, taluka, season, irrigated, top_k, None)


def _recommend(district, taluka, season, irrigated, top_k, soil_test) -> dict:
    if season not in config.APY_SEASONS:
        raise HTTPException(400, f"season must be one of {config.APY_SEASONS}")
    test = None
    if soil_test is not None and any(v is not None for v in soil_test.model_dump().values()):
        test = SoilTest(**soil_test.model_dump())
    try:
        rec = recommend(district, taluka, season, soil_test=test,
                        irrigated=irrigated, top_k=top_k)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return asdict(rec)
