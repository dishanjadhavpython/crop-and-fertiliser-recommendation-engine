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
@lru_cache(maxsize=None)
def _table() -> pd.DataFrame:
    """Clean inorganic rows with a numeric quantity — the per-hectare lookup."""
    f = load_fertiliser()
    f = f[(f["Category"] == "Inorganic") & (f["Unit"] == "Kg per Hectare")].copy()
    f["Quantity"] = pd.to_numeric(f["Quantity"], errors="coerce")
    return f.dropna(subset=["Quantity", "Soil_Class", "Option"]).reset_index(drop=True)


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


def lookup(
    district: str,
    crop: str,
    soil_class: str,
    *,
    variety: str | None = None,
    irrigation: str | None = None,
    season: str | None = None,
    option: int = 1,
) -> dict | None:
    """Layer 2 — the exact table lookup. Returns products and nutrient target.

    Unspecified context dimensions resolve to the table's single published
    value when there is exactly one, so a caller who knows only
    (district, crop, class) still gets the right row.
    """
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
        if want is not None:
            sub = sub[sub[col].astype("string").str.casefold() == want.casefold()]
        elif sub[col].nunique(dropna=False) > 1:
            # ambiguous and unspecified: prefer the rainfed / all-variety default
            default = _default_for(col, sub)
            sub = sub[sub[col].astype("string").fillna("") == (default or "")]
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
    """The nutrient target at each of the three fertility classes."""
    out = {}
    for cls in config.SOIL_CLASSES:
        hit = lookup(district, crop, cls, **ctx)
        if hit:
            out[cls] = hit["target"]
    return out


# ------------------------------------------------------ micronutrients L4 ----
def micronutrient_plan(taluka_features: dict) -> list[dict]:
    """Layer 4 — corrections for the six components the table ignores.

    The government recommendation uses N, P, K and OC. The Soil Health Card
    carries twelve components; sulphur, iron, zinc, copper, boron and manganese
    go completely unused despite deficiency rates that vary by up to 34
    percentage points between talukas. Closing that gap is real agronomy.
    """
    plan = []
    for comp in config.MICRONUTRIENTS:
        deficient_pct = float(taluka_features.get(f"def_{comp}", 0.0))
        if deficient_pct <= config.MICRO_DEFICIENCY_TRIGGER_PCT:
            continue
        spec = MICRONUTRIENT_CORRECTIONS[comp]
        entry = {
            "component": comp,
            "deficient_pct": round(deficient_pct, 1),
            "critical_limit": spec["critical"],
            "product": spec["product"],
            "rate_kg_ha": spec["rate_kg_ha"],
            "note": spec["note"],
            "priority": "high" if deficient_pct > 70 else "moderate",
        }
        if comp == "Zn":
            lockout = float(taluka_features.get("zn_lockout", 0.0))
            if lockout > 20:
                entry["note"] += (f" Alkaline lockout is compounding here "
                                  f"(zn_lockout={lockout:.0f}); prefer a chelated source.")
        plan.append(entry)
    return sorted(plan, key=lambda e: -e["deficient_pct"])


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
    **ctx,
) -> Recommendation | None:
    """Full S4 pipeline for one crop.

    With ``interpolate=False`` this returns the government table verbatim —
    the mode the exactness test asserts against.
    """
    from src.rules.soil_class import taluka_soil_test

    test = soil_test or taluka_soil_test(taluka_features)
    soil_class = classify(test)

    exact = lookup(district, crop, soil_class, **ctx)
    if exact is None:
        return None

    rec = Recommendation(crop=exact["crop"], district=exact["district"],
                         soil_class=soil_class, exact_table=exact)

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

    rec.micronutrients = micronutrient_plan(taluka_features)
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
            "Soil_Class", "Option", "Fertilizer"]
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
) -> dict | None:
    """State-wide median recipe, for crop-district pairs the table omits.

    Explicitly an estimate, never presented as the published recommendation.
    """
    m = _state_medians()
    sub = m[(m["Crop"].str.casefold() == crop.casefold())
            & (m["Soil_Class"] == soil_class)
            & (m["Option"] == option)]
    if sub.empty:
        return None

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
