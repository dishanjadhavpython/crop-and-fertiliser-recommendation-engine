"""The contract with the AgroSense frontend, checked from this side.

`tests/test_frontend.py` guards the console this repo serves. This guards the
*other* consumer: the Next app one directory up, which renders the same
response through its own TypeScript types.

Two of these are cross-repo on purpose, and they are the ones that matter.
`scripts/check-ontology.mjs` in the app already fails the build when somebody
edits `cropOntology.ts` and leaves a crop unnamed — but it reads only the app's
own files, so it cannot see this repo growing a nineteenth crop into a
twentieth. That failure would be silent and would reach a farmer as a
recommendation card with a raw APY string on it (`Arhar/Tur`, `Cotton(lint)`)
and a detail link that 404s, because all three `/prediction/*` routes set
`dynamicParams = false`.

So the engine asserts, from here, that everything it can emit the app can name.

Skipped rather than failed when the app is not present, so this repo still
stands alone.
"""
import json
import re
import warnings
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data.load import load_soil_type
from src.serve.api import app

warnings.filterwarnings("ignore")

APP = Path(__file__).resolve().parents[2]
ONTOLOGY = APP / "src" / "data" / "cropOntology.ts"
CROPS_TS = APP / "src" / "data" / "crops.ts"

needs_app = pytest.mark.skipif(
    not ONTOLOGY.exists(), reason="the Next app is not checked out beside this repo"
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _ts_map(const: str) -> dict[str, str]:
    """Pull a `Record<string, string>` literal out of the TypeScript source.

    Parsed rather than imported because there is no TypeScript runtime here,
    and a duplicated copy of the mapping in Python would be the very drift
    this test exists to catch.
    """
    block = re.search(
        rf"{const}: Record<string, string> = \{{(.*?)\n\}};",
        ONTOLOGY.read_text(encoding="utf-8"),
        re.S,
    )
    assert block, f"{const} not found in {ONTOLOGY.name}"
    pairs = re.findall(
        r'^\s*(?:"([^"]+)"|([A-Za-z_$][\w$]*))\s*:\s*"([^"]+)"',
        block.group(1),
        re.M,
    )
    return {(q or b): v for q, b, v in pairs}


def _app_crop_keys() -> set[str]:
    return set(re.findall(r'\bcrop\("([^"]+)"', CROPS_TS.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# The vocabularies
# --------------------------------------------------------------------------


@needs_app
def test_every_crop_this_engine_can_rank_has_a_name_in_the_app():
    from src.ontology.crop_map import CROP_ONTOLOGY

    mapping, keys = _ts_map("ENGINE_TO_CROP_KEY"), _app_crop_keys()
    missing = sorted(set(CROP_ONTOLOGY) - set(mapping))
    assert not missing, (
        f"the app cannot name {missing}. Add them to ENGINE_TO_CROP_KEY in "
        f"cropOntology.ts, to CROPS in crops.ts, and reserve their image slot "
        f"in assets.ts."
    )
    dangling = sorted(k for k in mapping.values() if k not in keys)
    assert not dangling, f"cropOntology.ts points at crops that do not exist: {dangling}"


@needs_app
def test_every_fertiliser_product_this_engine_doses_has_a_card_in_the_app():
    """The table prescribes four straight fertilisers; the app must stock all four.

    MOP and SSP were missing when this integration started, so a real answer of
    "88.67 kg/ha of MOP" had no product to point at.
    """
    import pandas as pd

    from src import config

    table = pd.read_csv(config.F_FERT)
    inorganic = set(
        table.loc[table["Category"] == "Inorganic", "Fertilizer"].dropna().unique()
    )
    mapping = _ts_map("ENGINE_TO_FERTILIZER_KEY")
    missing = sorted(inorganic - set(mapping))
    assert not missing, f"the app has no card for {missing}"


# --------------------------------------------------------------------------
# The response shape the TypeScript declares
# --------------------------------------------------------------------------

#: `Recommendation` in `src/lib/recommendTypes.ts`.
TOP_LEVEL = {
    "district", "taluka", "season", "irrigated", "soil_class",
    "soil_test_source", "confident", "novelty", "abstention_reason",
    "water_limited", "crops", "vetoed", "not_assessable", "micronutrients",
    "context",
}

#: `CropAdvice`.
CROP_FIELDS = {
    "crop", "crop_marathi", "rank", "final_score", "learned_score",
    "rule_score", "suitability_class", "limiting_factor", "decided_by",
    "reason", "yield_p10_t_ha", "yield_p50_t_ha", "yield_p90_t_ha",
    "yield_interval_note", "yield_class", "yield_class_confidence",
    "yield_regime", "yield_abstained", "factors", "requires_irrigation",
    "evidence", "fertiliser",
}

#: `RecommendContext`. `surveyed_soil` is the one the soil-photo check reads.
CONTEXT_FIELDS = {
    "annual_rainfall_mm", "season_rainfall_mm", "rootzone_awc_mm",
    "soil_depth_mm", "drainage_ord", "aridity_index", "lgp_days",
    "leach_risk", "soil_test", "ph_source", "ph_used", "ec_source",
    "ec_saline_pct_used", "shc_samples", "surveyed_soil",
}


def test_response_carries_every_field_the_typescript_declares(client):
    d = client.post("/recommend", json={
        "district": "SOLAPUR", "taluka": "SANGOLE", "season": "Rabi", "top_k": 5,
    }).json()

    assert TOP_LEVEL <= set(d), f"missing: {sorted(TOP_LEVEL - set(d))}"
    assert CONTEXT_FIELDS <= set(d["context"]), (
        f"missing from context: {sorted(CONTEXT_FIELDS - set(d['context']))}"
    )
    for crop in d["crops"]:
        assert CROP_FIELDS <= set(crop), (
            f"{crop['crop']} missing: {sorted(CROP_FIELDS - set(crop))}"
        )


def test_surveyed_soil_is_description_the_photo_check_can_use(client):
    """`SoilAgreement` in the app compares a photo class against these words."""
    s = client.post("/recommend", json={
        "district": "SOLAPUR", "taluka": "SANGOLE", "season": "Rabi",
    }).json()["context"]["surveyed_soil"]

    assert s is not None
    assert set(s) == {
        "soil_type", "soil_type_secondary", "share_pct", "texture",
        "depth", "drainage", "parent_material", "points_sampled",
    }
    # The app maps five of the classifier's eight classes onto this vocabulary;
    # a value outside it would silently become "unmapped" for every photo.
    assert s["soil_type"] in set(load_soil_type()["Soil_Type"].dropna().unique())


# --------------------------------------------------------------------------
# Smoke: one taluka per soil type, nothing nameless
# --------------------------------------------------------------------------


def _one_taluka_per_soil_type() -> list[tuple[str, str, str]]:
    df = load_soil_type()
    df = df[~df["urban_no_shc"]]
    out = []
    for soil_type, group in df.groupby("Soil_Type"):
        row = group.iloc[0]
        out.append((str(soil_type), str(row["District"]), str(row["Taluka"])))
    return sorted(out)


@needs_app
@pytest.mark.parametrize("soil_type,district,taluka", _one_taluka_per_soil_type())
def test_no_crop_comes_back_without_a_name(client, soil_type, district, taluka):
    """Across every soil type in the state, nothing the app cannot label.

    Ranked crops, vetoed crops and the fertiliser products inside the dose
    plan are all rendered by name, so all three have to resolve.
    """
    r = client.post("/recommend", json={
        "district": district, "taluka": taluka, "season": "Kharif", "top_k": 5,
    })
    if r.status_code == 404:
        pytest.skip(f"{district}/{taluka} is outside the 351 SHC talukas")
    assert r.status_code == 200, r.text
    d = r.json()

    crops = _ts_map("ENGINE_TO_CROP_KEY")
    ferts = _ts_map("ENGINE_TO_FERTILIZER_KEY")

    for advice in d["crops"]:
        assert advice["crop"] in crops, f"{soil_type}: unnamed crop {advice['crop']!r}"
        plan = advice.get("fertiliser") or {}
        for product in (plan.get("table_products_kg_ha") or {}):
            assert product in ferts, f"{soil_type}: unnamed product {product!r}"

    for v in d["vetoed"]:
        assert v["crop"] in crops, f"{soil_type}: unnamed vetoed crop {v['crop']!r}"
