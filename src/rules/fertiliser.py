"""S4 — the fertiliser engine (plan §6). Deterministic, never learned.

Across all 41,070 rows the number of keys carrying more than one distinct
quantity is **zero**: the table is an exact function. A model fitted to it can
only approximate what is already perfect, so this is a lookup with four layers
of value added on top.

    L1  soil test -> fertility class          src/rules/soil_class.py
    L2  exact lookup against the table        recommend(..., interpolate=False)
    L3  interpolation between the three archetypes
    L4  micronutrient correction  <- the contribution the table is missing
    L5  split scheduling driven by leach_risk
    (L6 least-cost blend lives in cost_optimiser.py)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_fertiliser
from src.rules.soil_class import SoilTest, classify

#: the three nutrients the government table prescribes
NUTRIENTS = ["N", "P2O5", "K2O"]

#: nutrient content of each straight/complex product, as a mass fraction
PRODUCT_ANALYSIS: dict[str, dict[str, float]] = {
    "Urea": {"N": 0.46},
    "DAP": {"N": 0.18, "P2O5": 0.46},
    "SSP": {"P2O5": 0.16, "S": 0.12},     # SSP carries ~12% S — see the sulphur note in §6.3
    "MOP": {"K2O": 0.60},
}

#: indicative Maharashtra retail prices, INR per kg (subsidised rates, 2024-25).
#: Overridable at call time; the LP in cost_optimiser reads these by default.
PRODUCT_PRICE_INR_PER_KG: dict[str, float] = {
    "Urea": 5.36,
    "DAP": 27.00,
    "SSP": 9.50,
    "MOP": 34.00,
}

#: L4 — micronutrient corrections (plan §6.3). Critical limits are the ICAR
#: values; the trigger here is the taluka's *deficient sample share*.
MICRONUTRIENT_CORRECTIONS: dict[str, dict] = {
    "S": {
        "critical": "< 10 ppm",
        "product": "Gypsum",
        "rate_kg_ha": 200.0,
        "note": "Or substitute SSP for DAP — SSP carries ~12% S, so the "
                "correction costs nothing extra.",
    },
    "Zn": {
        "critical": "< 0.6 ppm",
        "product": "ZnSO4.7H2O",
        "rate_kg_ha": 25.0,
        "note": "Soil-applied. Worsens on alkaline soils — see zn_lockout.",
    },
    "Fe": {
        "critical": "< 4.5 ppm",
        "product": "FeSO4",
        "rate_kg_ha": 25.0,
        "note": "Foliar 0.5% spray is more effective on calcareous soils.",
    },
    "B": {
        "critical": "< 0.5 ppm",
        "product": "Borax",
        "rate_kg_ha": 10.0,
        "note": "Narrow safety margin — never exceed; toxicity is a real risk.",
    },
    "Mn": {
        "critical": "< 2.0 ppm",
        "product": "MnSO4",
        "rate_kg_ha": 20.0,
        "note": "Often co-occurs with zinc deficiency.",
    },
    "Cu": {
        "critical": "< 0.2 ppm",
        "product": "CuSO4",
        "rate_kg_ha": 10.0,
        "note": "Rare in Maharashtra — 98% of talukas are sufficient.",
    },
}

#: leach_risk percentile above which nitrogen is split more finely (L5)
LEACH_SPLIT_THRESHOLDS = (0.60, 0.85)


# ---------------------------------------------------------------- lookup ----
#: The table's two context columns are used interchangeably: ``Crop_Season``
#: carries "Irrigated"/"Rainfed" on 1,672 rows, and ``Crop_Irrigation`` carries
#: variety-ish values such as "BT" and "1st year". So the season and the water
#: regime are read out of *both* columns by vocabulary rather than by position.
SEASON_TAGS = {"kharif": "Kharif", "pre-kharif": "Kharif", "rabi": "Rabi",
               "summer": "Summer", "annual crop": "Whole Year"}
WATER_TAGS = {"irrigated": "irrigated", "rainfed": "rainfed"}


def _context_tags(irrigation, season) -> tuple[str | None, str | None]:
    """(season, water regime) a published row actually states, or None."""
    words = [str(v).strip().casefold() for v in (irrigation, season) if _norm(v) is not None]
    return (next((SEASON_TAGS[w] for w in words if w in SEASON_TAGS), None),
            next((WATER_TAGS[w] for w in words if w in WATER_TAGS), None))


@lru_cache(maxsize=None)
def _table() -> pd.DataFrame:
    """Clean inorganic rows with a numeric quantity — the per-hectare lookup."""
    f = load_fertiliser()
    f = f[(f["Category"] == "Inorganic") & (f["Unit"] == "Kg per Hectare")].copy()
    f["Quantity"] = pd.to_numeric(f["Quantity"], errors="coerce")
    out = f.dropna(subset=["Quantity", "Soil_Class", "Option"]).reset_index(drop=True)
    tags = [_context_tags(i, s) for i, s in zip(out["Crop_Irrigation"], out["Crop_Season"])]
    out["ctx_season"] = [t[0] for t in tags]
    out["ctx_water"] = [t[1] for t in tags]
    return out


@lru_cache(maxsize=None)
def _index() -> dict:
    """(district, crop, variety, irrigation, season, class, option) -> products."""
    idx: dict[tuple, dict] = {}
    for row in _table().itertuples(index=False):
        key = (
            row.District, row.Crop, row.Crop_Variety,
            _norm(row.Crop_Irrigation), _norm(row.Crop_Season),
            row.Soil_Class, int(row.Option),
        )
        entry = idx.setdefault(key, {"products": {}, "target": {}})
        entry["products"][row.Fertilizer] = float(row.Quantity)
        entry["target"] = {
            "N": _f(row.N_kg_per_ha),
            "P2O5": _f(row.P2O5_kg_per_ha),
            "K2O": _f(row.K2O_kg_per_ha),
        }
    return idx


def _norm(v) -> str | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v)


def _f(v) -> float | None:
    return None if pd.isna(v) else float(v)


def available_contexts(district: str, crop: str) -> pd.DataFrame:
    """The variety / irrigation / season combinations the table publishes."""
    t = _table()
    sub = t[(t["District"] == district.upper()) & (t["Crop"].str.casefold() == crop.casefold())]
    cols = ["Crop_Variety", "Crop_Irrigation", "Crop_Season", "Soil_Class", "Option"]
    return sub[cols].drop_duplicates().reset_index(drop=True)


def resolve_context(
    district: str,
    crop: str,
    *,
    want_season: str | None = None,
    want_irrigated: bool | None = None,
    variety: str | None = None,
) -> dict | None:
    """Which published context answers *this farmer's* season and water regime.

    The table often publishes several recipes for one crop in one district —
    Kharif and Rabi, irrigated and rainfed — and they differ by more than
    rounding. Serving used to ignore the request entirely and fall back to
    "Rainfed, then Kharif", so an irrigated Rabi farmer could be handed the
    rainfed Kharif dose. This picks the row that states what was asked for,
    preferring a context that names the requested season and water regime over
    one that names neither, and never one that names a different season.
    """
    t = _table()
    sub = t[(t["District"] == district.upper())
            & (t["Crop"].str.casefold() == crop.casefold())]
    if variety is not None:
        sub = sub[sub["Crop_Variety"].astype("string").str.casefold() == variety.casefold()]
    if sub.empty:
        return None

    want_water = None if want_irrigated is None else ("irrigated" if want_irrigated else "rainfed")
    want_season = SEASON_TAGS.get((want_season or "").strip().casefold(), want_season)

    def rank(row) -> tuple:
        # pd.isna, not `is None`: a context that states nothing arrives as NaN
        # out of the table, and treating that as "states something different"
        # made the generic recipe lose to a contradicting one.
        stated_season = None if pd.isna(row.ctx_season) else row.ctx_season
        stated_water = None if pd.isna(row.ctx_water) else row.ctx_water
        if want_season and stated_season == want_season:
            s = 2
        elif stated_season is None:
            s = 1                      # says nothing about season: still usable
        else:
            s = 0                      # names a different season
        if want_water and stated_water == want_water:
            w = 2
        elif stated_water is None:
            w = 1
        elif want_water is None and stated_water == "rainfed":
            w = 1                      # unasked: the table's own default
        else:
            w = 0
        variety_default = str(row.Crop_Variety).strip().casefold().startswith("all variety")
        # ``min(s, w)`` leads, so a context that CONTRADICTS the request on
        # either dimension loses to one that simply says nothing. Nagpur
        # sunflower is the case that settles it: the district publishes an
        # "Irrigated Kharif" recipe and a general one, and a rainfed farmer
        # should be given the general recipe rather than the irrigated dose.
        return min(s, w), s + w, s, w, variety_default

    combos = sub[["Crop_Variety", "Crop_Irrigation", "Crop_Season",
                  "ctx_season", "ctx_water"]].drop_duplicates()
    best = max(combos.itertuples(index=False), key=rank)
    return {"variety": _norm(best.Crop_Variety),
            "irrigation": _norm(best.Crop_Irrigation),
            "season": _norm(best.Crop_Season),
            "states_season": best.ctx_season,
            "states_water": best.ctx_water}


def _narrow(sub: pd.DataFrame, col: str, want, strict: bool) -> pd.DataFrame:
    if want is not None:
        return sub[sub[col].astype("string").str.casefold() == want.casefold()]
    if strict:
        return sub[sub[col].isna()]
    if sub[col].nunique(dropna=False) > 1:
        # ambiguous and unspecified: prefer the rainfed / all-variety default
        default = _default_for(col, sub)
        return sub[sub[col].astype("string").fillna("") == (default or "")]
    return sub


def lookup(
    district: str,
    crop: str,
    soil_class: str,
    *,
    variety: str | None = None,
    irrigation: str | None = None,
    season: str | None = None,
    option: int = 1,
    want_season: str | None = None,
    want_irrigated: bool | None = None,
) -> dict | None:
    """Layer 2 — the exact table lookup. Returns products and nutrient target.

    ``variety`` / ``irrigation`` / ``season`` name the table's own column values
    and are matched verbatim — that is what the exactness test asserts against.
    ``want_season`` / ``want_irrigated`` instead describe the *farmer's* query
    and are resolved to a published context by ``resolve_context``.
    """
    strict = False
    if irrigation is None and season is None and (want_season or want_irrigated is not None):
        ctx = resolve_context(district, crop, want_season=want_season,
                              want_irrigated=want_irrigated, variety=variety)
        if ctx:
            variety = variety if variety is not None else ctx["variety"]
            irrigation, season = ctx["irrigation"], ctx["season"]
            strict = True              # pin the resolved row, NaNs included

    t = _table()
    sub = t[(t["District"] == district.upper()) & (t["Crop"].str.casefold() == crop.casefold())]
    if sub.empty:
        return None
    sub = sub[(sub["Soil_Class"] == soil_class) & (sub["Option"] == option)]
    if sub.empty:
        return None

    for col, want in (("Crop_Variety", variety),
                      ("Crop_Irrigation", irrigation),
                      ("Crop_Season", season)):
        sub = _narrow(sub, col, want, strict)
        if sub.empty:
            return None

    row = sub.iloc[0]
    return {
        "district": row["District"],
        "crop": row["Crop"],
        "variety": row["Crop_Variety"],
        "irrigation": _norm(row["Crop_Irrigation"]),
        "season": _norm(row["Crop_Season"]),
        "soil_class": row["Soil_Class"],
        "option": int(row["Option"]),
        "products": dict(zip(sub["Fertilizer"], sub["Quantity"].astype(float))),
        "target": {"N": _f(row["N_kg_per_ha"]),
                   "P2O5": _f(row["P2O5_kg_per_ha"]),
                   "K2O": _f(row["K2O_kg_per_ha"])},
    }


def _default_for(col: str, sub: pd.DataFrame) -> str | None:
    """Pick a deterministic default when a context dimension is unspecified."""
    values = sub[col].dropna().unique().tolist()
    preference = {
        "Crop_Irrigation": ["Rainfed", "Irrigated"],
        "Crop_Variety": ["All variety", "All Variety"],
        "Crop_Season": ["Kharif", "Rabi", "Summer"],
    }[col]
    for p in preference:
        if p in values:
            return p
    return sorted(values)[0] if values else None


# --------------------------------------------------------- interpolation ----
def interpolate_target(crop_targets: dict[str, dict], test: SoilTest) -> dict[str, float]:
    """Layer 3 — interpolate the nutrient target between the three archetypes.

    Hard bucketing creates a cliff: a farmer at 279 kg N/ha gets a materially
    different recommendation from one at 281. Since the table publishes the
    three archetype anchors, interpolating on the farmer's actual value is
    standard STCR practice and a clean improvement on the table's own behaviour.

    ``crop_targets`` maps 'Low'/'Medium'/'High' -> {'N': .., 'P2O5': .., 'K2O': ..}.
    Each nutrient is driven by its own soil component, independently.
    """
    driver = {"N": "N", "P2O5": "P", "K2O": "K"}
    measured = test.as_dict()
    out: dict[str, float] = {}

    for nutrient, component in driver.items():
        anchors = config.SOIL_ARCHETYPES[component]
        value = measured.get(component)
        doses = {c: crop_targets.get(c, {}).get(nutrient) for c in config.SOIL_CLASSES}
        if value is None or any(d is None for d in doses.values()):
            out[nutrient] = doses.get("Medium") or 0.0
            continue

        lo_x, mid_x, hi_x = anchors["Low"], anchors["Medium"], anchors["High"]
        # clamp outside the anchor span rather than extrapolating
        x = min(max(float(value), lo_x), hi_x)
        if x <= mid_x:
            frac = (x - lo_x) / (mid_x - lo_x)
            out[nutrient] = doses["Low"] + (doses["Medium"] - doses["Low"]) * frac
        else:
            frac = (x - mid_x) / (hi_x - mid_x)
            out[nutrient] = doses["Medium"] + (doses["High"] - doses["Medium"]) * frac
    return out


def crop_targets(district: str, crop: str, **ctx) -> dict[str, dict]:
    """The nutrient target at each of the three fertility classes.

    All three classes are resolved in the **same** context. Resolving each one
    separately let Low come from the rainfed row and High from the irrigated
    one, so interpolating between them crossed two different recommendations.
    """
    out = {}
    for cls in config.SOIL_CLASSES:
        hit = lookup(district, crop, cls, **ctx)
        if hit:
            out[cls] = hit["target"]
    return out


# ------------------------------------------------------ micronutrients L4 ----
def micronutrient_plan(
    taluka_features: dict, farmer_micro: dict[str, str] | None = None
) -> list[dict]:
    """Layer 4 — corrections for the six components the table ignores.

    The government recommendation uses N, P, K and OC. The Soil Health Card
    carries twelve components; sulphur, iron, zinc, copper, boron and manganese
    go completely unused despite deficiency rates that vary by up to 34
    percentage points between talukas. Closing that gap is real agronomy.

    ``farmer_micro`` is the farmer's own card, keyed by the same short codes as
    ``config.MICRONUTRIENTS``, valued "low"/"normal"/"high". A component with a
    farmer verdict is decided by that verdict alone — a real reading from this
    field outranks a taluka-wide average — and the taluka's ``def_{comp}`` share
    is consulted only for components the card is silent on.
    """
    farmer_micro = farmer_micro or {}
    plan = []
    for comp in config.MICRONUTRIENTS:
        farmer_status = farmer_micro.get(comp)
        if farmer_status is not None:
            if farmer_status != "low":
                continue
            source, deficient_pct, priority = "farmer soil health card", None, "high"
        else:
            deficient_pct = float(taluka_features.get(f"def_{comp}", 0.0))
            if deficient_pct <= config.MICRO_DEFICIENCY_TRIGGER_PCT:
                continue
            source = "taluka SHC distribution"
            priority = "high" if deficient_pct > 70 else "moderate"

        spec = MICRONUTRIENT_CORRECTIONS[comp]
        entry = {
            "component": comp,
            "source": source,
            "deficient_pct": round(deficient_pct, 1) if deficient_pct is not None else None,
            "critical_limit": spec["critical"],
            "product": spec["product"],
            "rate_kg_ha": spec["rate_kg_ha"],
            "note": spec["note"],
            "priority": priority,
        }
        if comp == "Zn":
            lockout = float(taluka_features.get("zn_lockout", 0.0))
            if lockout > 20:
                entry["note"] += (f" Alkaline lockout is compounding here "
                                  f"(zn_lockout={lockout:.0f}); prefer a chelated source.")
        plan.append(entry)
    # A farmer-confirmed deficiency sorts first regardless of taluka share —
    # it is the more certain statement, not a smaller one.
    return sorted(plan, key=lambda e: (e["deficient_pct"] is not None, -(e["deficient_pct"] or 0)))


def sulphur_substitution(products: dict[str, float], micro: list[dict]) -> dict | None:
    """The free sulphur correction: swap DAP for SSP when S is deficient.

    SSP is ~16% P2O5 and ~12% S against DAP's 46% P2O5 and no sulphur, so the
    swap needs ~2.9x the mass but corrects sulphur at no extra nutrient cost.
    """
    if not any(m["component"] == "S" for m in micro) or "DAP" not in products:
        return None
    dap_kg = products["DAP"]
    p2o5 = dap_kg * PRODUCT_ANALYSIS["DAP"]["P2O5"]
    ssp_kg = p2o5 / PRODUCT_ANALYSIS["SSP"]["P2O5"]
    return {
        "swap": "DAP -> SSP",
        "dap_kg_ha": round(dap_kg, 1),
        "ssp_kg_ha": round(ssp_kg, 1),
        "sulphur_supplied_kg_ha": round(ssp_kg * PRODUCT_ANALYSIS["SSP"]["S"], 1),
        "nitrogen_shortfall_kg_ha": round(dap_kg * PRODUCT_ANALYSIS["DAP"]["N"], 1),
        "note": "Top up the lost DAP nitrogen with urea; the sulphur then costs nothing.",
    }


# ------------------------------------------------------------- split L5 ----
def split_schedule(n_kg_ha: float, leach_risk: float, leach_percentile: float,
                   crop: str = "") -> dict:
    """Layer 5 — split the nitrogen dose according to leaching risk.

    The table gives a total, not a schedule. Where high rainfall, coarse
    texture and low organic carbon combine, applied nitrogen is lost before the
    crop can use it, so it is split more finely. Phosphorus and potassium stay
    as a single basal dose.
    """
    if leach_percentile >= LEACH_SPLIT_THRESHOLDS[1]:
        fracs, timing, band = [0.25, 0.25, 0.25, 0.25], \
            ["basal", "20-25 DAS", "40-45 DAS", "60-65 DAS"], "high"
    elif leach_percentile >= LEACH_SPLIT_THRESHOLDS[0]:
        fracs, timing, band = [0.34, 0.33, 0.33], ["basal", "25-30 DAS", "50-55 DAS"], "moderate"
    else:
        fracs, timing, band = [0.5, 0.5], ["basal", "30-35 DAS"], "low"

    return {
        "leach_risk": round(float(leach_risk), 2),
        "leach_risk_band": band,
        "n_splits": len(fracs),
        "schedule": [
            {"stage": t, "n_kg_ha": round(n_kg_ha * f, 1)} for t, f in zip(timing, fracs)
        ],
        "phosphorus_potassium": "full dose as basal",
        "rationale": {
            "low": "Nitrogen is retained; two applications are sufficient.",
            "moderate": "Moderate leaching — three splits reduce loss.",
            "high": "High leaching risk from rainfall, coarse texture and low "
                    "organic carbon; four splits keep nitrogen in the root zone.",
        }[band],
    }


# --------------------------------------------------------------- top level ---
@dataclass
class Recommendation:
    crop: str
    district: str
    soil_class: str
    #: the published context this dose came from — season, water regime, variety
    context: dict = field(default_factory=dict)
    exact_table: dict = field(default_factory=dict)
    interpolated_target: dict = field(default_factory=dict)
    micronutrients: list = field(default_factory=list)
    sulphur_swap: dict | None = None
    schedule: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def recommend(
    district: str,
    crop: str,
    taluka_features: dict,
    soil_test: SoilTest | None = None,
    *,
    interpolate: bool = True,
    leach_percentile: float = 0.5,
    want_season: str | None = None,
    want_irrigated: bool | None = None,
    **ctx,
) -> Recommendation | None:
    """Full S4 pipeline for one crop.

    With ``interpolate=False`` this returns the government table verbatim —
    the mode the exactness test asserts against. ``want_season`` and
    ``want_irrigated`` carry the farmer's query; the context they resolve to is
    pinned once here and used for every class, so the interpolation stays
    inside one published recommendation.
    """
    from src.rules.soil_class import taluka_soil_test

    test = soil_test or taluka_soil_test(taluka_features)
    soil_class = classify(test)

    resolved = None
    if not ctx.get("irrigation") and not ctx.get("season") and (
            want_season or want_irrigated is not None):
        resolved = resolve_context(district, crop, want_season=want_season,
                                   want_irrigated=want_irrigated,
                                   variety=ctx.get("variety"))
        if resolved:
            ctx = {**ctx, "variety": ctx.get("variety") or resolved["variety"],
                   "irrigation": resolved["irrigation"], "season": resolved["season"]}

    exact = lookup(district, crop, soil_class, **ctx)
    if exact is None:
        return None

    rec = Recommendation(crop=exact["crop"], district=exact["district"],
                         soil_class=soil_class, exact_table=exact,
                         context={"season": exact["season"],
                                  "irrigation": exact["irrigation"],
                                  "variety": exact["variety"],
                                  "states_season": (resolved or {}).get("states_season"),
                                  "states_water": (resolved or {}).get("states_water")})

    if interpolate:
        targets = crop_targets(district, crop, **ctx)
        if len(targets) == 3:
            rec.interpolated_target = {
                k: round(v, 1) for k, v in interpolate_target(targets, test).items()
            }
            rec.notes.append(
                "Nutrient target interpolated between the table's Low/Medium/High "
                "archetypes on the measured soil values (STCR practice), rather "
                "than bucketed — this removes the 279 vs 281 kg N/ha cliff."
            )
        else:
            rec.interpolated_target = dict(exact["target"])
            rec.notes.append("Only one fertility class published; interpolation skipped.")
    else:
        rec.interpolated_target = dict(exact["target"])

    rec.micronutrients = micronutrient_plan(taluka_features, farmer_micro=test.micronutrients)
    rec.sulphur_swap = sulphur_substitution(exact["products"], rec.micronutrients)

    n_target = rec.interpolated_target.get("N") or 0.0
    rec.schedule = split_schedule(
        n_target, taluka_features.get("leach_risk", 0.0), leach_percentile, crop
    )
    return rec


# ------------------------------------------------- district coverage gap ----
# The table publishes a recipe for only 63.7% of (mapped crop x district)
# cells: linseed appears in 3 districts of 34, sesame in 9. Doses genuinely
# vary across districts (55.5% of crop-context keys have more than one
# distinct quantity), so silently borrowing a neighbour's recipe would be
# wrong. Instead the state-wide median is offered, labelled as an estimate and
# carrying its own observed spread so the uncertainty is visible.
@lru_cache(maxsize=None)
def _state_medians() -> pd.DataFrame:
    t = _table()
    keys = ["Crop", "Crop_Variety", "Crop_Irrigation", "Crop_Season",
            "ctx_season", "ctx_water", "Soil_Class", "Option", "Fertilizer"]
    agg = t.groupby(keys, dropna=False).agg(
        qty_median=("Quantity", "median"),
        qty_min=("Quantity", "min"),
        qty_max=("Quantity", "max"),
        n_districts=("District", "nunique"),
        n_target=("N_kg_per_ha", "median"),
        p_target=("P2O5_kg_per_ha", "median"),
        k_target=("K2O_kg_per_ha", "median"),
    )
    return agg.reset_index()


def state_median_lookup(
    crop: str,
    soil_class: str,
    *,
    variety: str | None = None,
    irrigation: str | None = None,
    season: str | None = None,
    option: int = 1,
    want_season: str | None = None,
    want_irrigated: bool | None = None,
) -> dict | None:
    """State-wide median recipe, for crop-district pairs the table omits.

    Explicitly an estimate, never presented as the published recommendation.
    The farmer's season and water regime steer it exactly as they steer the
    district lookup — an estimate may be uncertain in magnitude, but it should
    not also be an answer to a different question.
    """
    m = _state_medians()
    sub = m[(m["Crop"].str.casefold() == crop.casefold())
            & (m["Soil_Class"] == soil_class)
            & (m["Option"] == option)]
    if sub.empty:
        return None

    if irrigation is None and season is None and (want_season or want_irrigated is not None):
        want_water = None if want_irrigated is None else (
            "irrigated" if want_irrigated else "rainfed")
        tag = SEASON_TAGS.get((want_season or "").strip().casefold(), want_season)
        match = sub[(sub["ctx_season"] == tag) | sub["ctx_season"].isna()] if tag else sub
        if want_water:
            tighter = match[(match["ctx_water"] == want_water) | match["ctx_water"].isna()]
            match = tighter if not tighter.empty else match
        if not match.empty:
            sub = match

    for col, want in (("Crop_Variety", variety),
                      ("Crop_Irrigation", irrigation),
                      ("Crop_Season", season)):
        if want is not None:
            sub = sub[sub[col].astype("string").str.casefold() == want.casefold()]
        elif sub[col].nunique(dropna=False) > 1:
            default = _default_for(col, sub)
            sub = sub[sub[col].astype("string").fillna("") == (default or "")]
        if sub.empty:
            return None

    row = sub.iloc[0]
    return {
        "crop": row["Crop"],
        "soil_class": soil_class,
        "option": int(row["Option"]),
        "products": dict(zip(sub["Fertilizer"], sub["qty_median"].astype(float))),
        "product_ranges": {
            f: (round(float(lo), 1), round(float(hi), 1))
            for f, lo, hi in zip(sub["Fertilizer"], sub["qty_min"], sub["qty_max"])
        },
        "target": {"N": _f(row["n_target"]), "P2O5": _f(row["p_target"]),
                   "K2O": _f(row["k_target"])},
        "n_districts": int(row["n_districts"]),
        "estimated": True,
        "note": (f"This district publishes no recommendation for {row['Crop']}. "
                 f"Shown is the state-wide median across {int(row['n_districts'])} "
                 f"districts that do — an estimate, not the official dose."),
    }


def crop_targets_state(crop: str, **ctx) -> dict[str, dict]:
    """State-median nutrient target at each fertility class."""
    out = {}
    for cls in config.SOIL_CLASSES:
        hit = state_median_lookup(crop, cls, **ctx)
        if hit:
            out[cls] = hit["target"]
    return out
