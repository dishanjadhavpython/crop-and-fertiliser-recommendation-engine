"""FastAPI serving layer (plan §9 week 7).

    uvicorn src.serve.api:app --reload

taluka + season + optional Soil Health Card values -> ranked crops with yield
bands, dose plans, and a plain-language reason per recommendation drawn from
the S2 limiting factor.

Marathi crop names come from the fertiliser table's own ``Crop_Local_Name``
column, which already carries them in Devanagari.
"""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.data.load import taluka_universe
from src.pipeline import load_pipeline, recommend
from src.rules.soil_class import SoilTest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Everything expensive, once, before the first farmer rather than during.

    `load_pipeline()` on its own was not enough. It restores the trained ranker
    from ``artifacts/pipeline_*.joblib`` in about a third of a second, but the
    first ``/recommend`` then spent a further ~4.6s inside
    ``build_feature_store()`` assembling the 351-taluka store from the raw
    CSVs — a cost paid by whoever happened to ask first. Warming both here
    moves it into container start, where the host's health check already
    allows for it, and every request including the first is served in ~0.13s.

    This replaces ``@app.on_event("startup")``, which FastAPI deprecated.
    """
    from src.features.build_store import build_feature_store

    load_pipeline()
    build_feature_store()
    yield


app = FastAPI(
    title="Crop & Fertiliser Recommendation Engine",
    description=__doc__,
    version="1.0",
    lifespan=lifespan,
)

#: Shared secret proving a request came from our own Next server rather than
#: from the internet. Read from the same variable the reading service uses
#: (`backend/config.py`), so one secret covers both processes.
#:
#: This engine has no authentication and no CORS headers of its own: it expects
#: to be reachable from the Next server and from nowhere else. Empty means
#: open, which is correct for a process bound to 127.0.0.1 and wrong for
#: anything else — set it before this ever has a public address.
#:
#: Note that setting it also closes the console at `/`, which cannot send the
#: header. That is the intended trade: the console is a development tool.
API_KEY = os.getenv("AGROSENSE_API_KEY", "").strip()


@app.middleware("http")
async def require_key(request: Request, call_next):
    """The network boundary, expressed in a header. Mirrors `backend/app.py`.

    `/health` stays open for the same reason the reading service's does: the
    host's own health check has no way to send the header, and a machine that
    fails its check is a machine the host restarts forever. It reports what is
    loaded, never anything read off a card.
    """
    if API_KEY and request.url.path != "/health":
        supplied = request.headers.get("x-agrosense-key") or ""
        if not secrets.compare_digest(supplied, API_KEY):
            return JSONResponse({"detail": "Not authorised."}, status_code=401)
    return await call_next(request)

#: SHC status verdict, the same three-way scale the card itself prints per
#: component ("low"/"normal"/"high" against the card's own printed range).
Status = Literal["low", "normal", "high"]

#: SoilTestIn field name -> the short code micronutrient_plan expects
#: (config.MICRONUTRIENTS).
MICRONUTRIENT_FIELDS: dict[str, str] = {
    "sulphur_status": "S", "zinc_status": "Zn", "iron_status": "Fe",
    "copper_status": "Cu", "boron_status": "B", "manganese_status": "Mn",
}


class SoilTestIn(BaseModel):
    """A farmer's own Soil Health Card values — all twelve components the card
    prints, all optional. N/P/K/OC drive the government fertility class; pH
    and EC replace the taluka average in the agronomic gate; the six
    micronutrient verdicts drive their own correction layer. See ``SoilTest``
    for what each does downstream.
    """
    n_kg_ha: float | None = Field(None, description="Available nitrogen, kg/ha")
    p_kg_ha: float | None = Field(None, description="Available phosphorus, kg/ha")
    k_kg_ha: float | None = Field(None, description="Available potassium, kg/ha")
    oc_pct: float | None = Field(None, description="Organic carbon, %")
    ph: float | None = Field(None, description="Measured soil pH")
    ec_status: Status | None = Field(
        None, description="EC verdict against the card's own printed range; 'high' = saline")
    sulphur_status: Status | None = Field(None, description="Available sulphur verdict")
    zinc_status: Status | None = Field(None, description="Available zinc verdict")
    iron_status: Status | None = Field(None, description="Available iron verdict")
    copper_status: Status | None = Field(None, description="Available copper verdict")
    boron_status: Status | None = Field(None, description="Available boron verdict")
    manganese_status: Status | None = Field(None, description="Available manganese verdict")


class RecommendIn(BaseModel):
    district: str
    taluka: str
    season: str = Field(..., description="Kharif | Rabi | Summer | Whole Year")
    irrigated: bool = False
    top_k: int = 5
    soil_test: SoilTestIn | None = None
    #: The soil-photo classifier's calibrated probabilities over its own
    #: classes, straight from `/api/soil`. The whole distribution, not the top
    #: few: fusion weighs it against the taluka survey, and a 0.45/0.44 split
    #: means something very different from a 0.45/0.05 one.
    soil_photo: dict[str, float] | None = Field(
        None, description="class -> probability, from the soil classifier")
    #: The caller's non-soil rejection. A photograph of a leaf or a hand must
    #: not update anything, and the classifier — not the engine — is what can
    #: tell. False makes fusion abstain.
    photo_in_distribution: bool = True


STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """The console. Served from the same origin as the API, so no CORS."""
    return FileResponse(STATIC / "index.html")


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
                      body.top_k, body.soil_test, body.soil_photo,
                      body.photo_in_distribution)


@app.get("/recommend")
def get_recommend(
    district: str,
    taluka: str,
    season: str = Query(..., description="Kharif | Rabi | Summer | Whole Year"),
    irrigated: bool = False,
    top_k: int = 5,
) -> dict:
    return _recommend(district, taluka, season, irrigated, top_k, None)


def _recommend(district, taluka, season, irrigated, top_k, soil_test,
               soil_photo=None, photo_in_distribution=True) -> dict:
    if season not in config.APY_SEASONS:
        raise HTTPException(400, f"season must be one of {config.APY_SEASONS}")
    test = None
    if soil_test is not None and any(v is not None for v in soil_test.model_dump().values()):
        micro = {
            code: getattr(soil_test, field)
            for field, code in MICRONUTRIENT_FIELDS.items()
            if getattr(soil_test, field) is not None
        }
        test = SoilTest(
            n_kg_ha=soil_test.n_kg_ha, p_kg_ha=soil_test.p_kg_ha,
            k_kg_ha=soil_test.k_kg_ha, oc_pct=soil_test.oc_pct,
            ph=soil_test.ph, ec_status=soil_test.ec_status,
            micronutrients=micro or None,
        )
    try:
        rec = recommend(district, taluka, season, soil_test=test,
                        irrigated=irrigated, top_k=top_k,
                        soil_photo=soil_photo,
                        photo_in_distribution=photo_in_distribution)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return asdict(rec)
