"""S2 — the agronomic gate and land-suitability scorer (plan §5.2).

Knowledge-based, no training data. That means it covers crops with no yield
record at all, and it is fully explainable.

Each factor scores 0-1, then the factors combine by **Liebig's law of the
minimum**, not by averaging. Averaging would let excellent rainfall mask a
fatal pH problem; the minimum is agronomically correct and is what makes the
gate trustworthy.

    score = min(f_rain, f_temp, f_pH, f_depth, f_drain, f_salinity, f_LGP, f_texture)

    S1 >= 0.75 highly suitable · S2 0.50-0.75 · S3 0.25-0.50 marginal · N < 0.25 unsuitable

Any crop scoring **N** is removed from the ranker's output regardless of its
learned score — a hard veto. Every veto is logged with its limiting factor;
that log is the explanation layer and the debugging tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config
from src.rules.crop_requirements import (
    NO_ENVELOPE,
    CropRequirement,
    Envelope,
    covered_crops,
    get,
)

SUITABILITY_CLASSES = [
    ("S1", 0.75, "highly suitable"),
    ("S2", 0.50, "suitable"),
    ("S3", 0.25, "marginally suitable"),
    ("N", 0.00, "not suitable"),
]

#: factors evaluated for every crop, in report order
FACTORS = ["rain", "temp", "pH", "depth", "drainage", "salinity", "LGP", "texture"]

#: Crops longer than this cannot complete inside a rainfed growing season
#: anywhere in Maharashtra (observed LGP spans 70-149 days), so the LGP factor
#: is not applied to them. Their water requirement is judged by effective_water.
LGP_MAX_DURATION_DAYS = 150

#: Factors that express a *physical* impossibility — no amount of irrigation
#: or management makes the land support the crop. Only these may veto.
HARD_FACTORS = frozenset({"depth", "drainage", "temp", "pH", "salinity", "season"})

#: Factors that express a water shortfall. A crop failing only on these is not
#: impossible here — it needs irrigation, which is a different statement and a
#: far more useful one.
WATER_FACTORS = frozenset({"rain", "LGP"})


def trapezoid(value: float, env: Envelope) -> float:
    """Score one factor on [0, 1] against a trapezoidal envelope."""
    if value is None or not np.isfinite(value):
        return 1.0                                   # unknown: do not penalise
    if value <= env.abs_min or value >= env.abs_max:
        return 0.0
    if env.opt_min <= value <= env.opt_max:
        return 1.0
    if value < env.opt_min:
        return (value - env.abs_min) / (env.opt_min - env.abs_min)
    return (env.abs_max - value) / (env.abs_max - env.opt_max)


def _depth_score(depth_mm: float, required_mm: float) -> float:
    """Full marks at or above the requirement, linear penalty below it.

    Half the required depth still yields a crop, but a quarter does not.
    """
    if depth_mm >= required_mm:
        return 1.0
    return max(0.0, (depth_mm / required_mm - 0.25) / 0.75)


def _salinity_score(saline_pct: float, tolerance_pct: float) -> float:
    """Share of saline samples against what the crop tolerates."""
    if saline_pct <= tolerance_pct:
        return 1.0
    # beyond tolerance, decay to zero at twice the tolerated share (min 10 pp)
    span = max(tolerance_pct, 10.0)
    return max(0.0, 1.0 - (saline_pct - tolerance_pct) / span)


def _lgp_score(lgp_days: float, required_days: float) -> float:
    if lgp_days >= required_days:
        return 1.0
    return max(0.0, (lgp_days / required_days - 0.4) / 0.6)


def _texture_score(texture_ord: float, preferred: tuple[str, ...]) -> float:
    """Soft preference, never a veto — texture rarely kills a crop outright."""
    if not preferred:
        return 1.0
    ord_map = {"Sandy": 1, "Sandy loam": 2, "Loamy-skeletal": 3,
               "Loamy": 4, "Clayey-skeletal": 5, "Clayey": 6}
    wanted = [ord_map[t] for t in preferred if t in ord_map]
    if not wanted or texture_ord is None or not np.isfinite(texture_ord):
        return 1.0
    distance = min(abs(texture_ord - w) for w in wanted)
    return float(max(0.6, 1.0 - 0.15 * distance))


def classify_score(score: float) -> tuple[str, str]:
    for code, floor, label in SUITABILITY_CLASSES:
        if score >= floor:
            return code, label
    return "N", "not suitable"


@dataclass
class Suitability:
    crop: str
    season: str
    score: float
    suitability_class: str
    label: str
    limiting_factor: str
    factors: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, float] = field(default_factory=dict)
    vetoed: bool = False
    reason: str = ""
    scored: bool = True
    #: score under the most favourable water scenario; the veto is decided on
    #: this, never on the rainfed score alone
    score_irrigated: float = float("nan")
    #: passes with irrigation but not without it — actionable, not a rejection
    requires_irrigation: bool = False
    #: which hard factor forced the veto, if any
    hard_limiting_factor: str | None = None


def _season_value(features: dict, base: str, season: str) -> float:
    """Read a seasonal feature, falling back to the annual value.

    Perennials ("Whole Year") are always evaluated on the annual aggregate.
    """
    if season in config.SEASONS:
        key = f"{base}_{season}"
        if key in features:
            return float(features[key])
    return float(features.get(f"{base}_annual", np.nan))


#: fraction of the root zone still holding usable water at the start of a season
RESIDUAL_MOISTURE_FRACTION = {"Kharif": 0.0, "Rabi": 1.0, "Summer": 0.3, "Whole Year": 0.0}


def effective_water(features: dict, season: str, irrigated: bool = False) -> float:
    """Water actually available to the crop over the season, in mm.

    In-season rainfall alone is the wrong test outside Kharif. Maharashtra's
    Rabi crops — chickpea, safflower, Rabi sorghum — are grown on the moisture
    the monsoon leaves stored in the profile, on soils that hold up to 270 mm.
    Scoring them against ~60 mm of Rabi rain would veto every Rabi crop in the
    state, which is plainly wrong.

        Kharif      rain as it falls
        Rabi        rain + the full root-zone store, recharged by the monsoon
        Summer      rain + 30% of the store, most of it already spent
        Whole Year  annual rainfall

    There is no irrigation variable in the data — the plan names it as the
    single biggest missing confounder (§7.4, §8 item 3) — so ``irrigated`` is
    an explicit caller-supplied switch rather than something inferred. Under
    it the farmer is assumed to supply the deficit up to the crop's optimum,
    and the water factor stops binding.
    """
    rain = _season_value(features, "rain", season)
    store = float(features.get("rootzone_awc", 0.0) or 0.0)
    return rain + RESIDUAL_MOISTURE_FRACTION.get(season, 0.0) * store


def score(crop: str, season: str, features: dict,
          irrigated: bool = False) -> Suitability:
    """Land-suitability score for one crop in one season at one taluka."""
    req = get(crop)
    if req is None:
        return Suitability(
            crop=crop, season=season, score=float("nan"),
            suitability_class="?", label="no envelope encoded",
            limiting_factor="none", scored=False,
            reason=("Aggregate or unencoded crop — the gate abstains rather than "
                    "inventing requirements."
                    if crop in NO_ENVELOPE else
                    "No requirement envelope encoded for this crop yet."),
        )

    # A crop grown outside its agronomic season is vetoed before any factor is
    # scored: sowing wheat in Kharif is a calendar error, not a marginal soil.
    if season not in req.seasons:
        return Suitability(
            crop=crop, season=season, score=0.0, suitability_class="N",
            label="not suitable", limiting_factor="season", vetoed=True,
            reason=f"{crop} is not grown in {season} "
                   f"(published seasons: {', '.join(req.seasons)}).",
        )

    rain = effective_water(features, season)
    tmax = _season_value(features, "tmax_mean", season)
    tmin = _season_value(features, "tmin_mean", season)
    tmean = (tmax + tmin) / 2.0
    lgp = _season_value(features, "lgp", season)
    ph = float(features.get("ph_class_value", np.nan))
    depth = float(features.get("depth_mm", np.nan))
    drainage = float(features.get("drainage_ord", np.nan))
    saline = float(features.get("ec_saline", 0.0))
    texture = float(features.get("texture_ord", np.nan))

    # Under irrigation the farmer tops the profile up to the crop's optimum,
    # so water stops being the binding constraint — but only upwards: excess
    # rainfall is still a real problem irrigation cannot undo.
    water_score = trapezoid(rain, req.rain_mm)
    if irrigated and rain < req.rain_mm.opt_min:
        water_score = 1.0

    factors = {
        "rain": water_score,
        "temp": trapezoid(tmean, req.temp_c),
        "pH": trapezoid(ph, req.ph),
        "depth": _depth_score(depth, req.min_depth_mm),
        "drainage": trapezoid(drainage, req.drainage),
        "salinity": _salinity_score(saline, req.max_saline_pct),
        # LGP counts days where rainfall exceeds half of ET0, so it measures
        # the *rainfed* growing season. Two limits follow, and the second was
        # a real bug: observed LGP across Maharashtra spans 70-149 days, so a
        # crop requiring 150 rainfed days could never pass anywhere. Sugarcane
        # was therefore vetoed in every taluka in the state, across 96% of the
        # area it actually occupies (79% of all LGP vetoes were false).
        #
        # A long-duration crop does not live on the rainfed season — it lives
        # on irrigation or stored profile moisture, which effective_water
        # already accounts for. So LGP gates only crops short enough to
        # complete within a rainfed season, and only in the rainfed season.
        "LGP": (_lgp_score(lgp, req.min_lgp_days)
                if (season in ("Kharif", "Whole Year")
                    and req.duration_days <= LGP_MAX_DURATION_DAYS)
                else 1.0),
        "texture": _texture_score(texture, req.preferred_textures),
    }

    # Liebig: the crop is only as good as its worst factor.
    limiting = min(factors, key=factors.get)
    total = factors[limiting]
    code, label = classify_score(total)

    # --- the veto decision -------------------------------------------------
    # A veto removes a crop from the recommendation entirely, so it must mean
    # "this land cannot support this crop", not "this land cannot support this
    # crop *without water*". Those are different statements and only the first
    # justifies a hard removal.
    #
    # Measured on eight years of practice, deciding the veto on the rainfed
    # score alone was wrong 38.7% of the time in Rabi and 27.6% in Summer —
    # both irrigated seasons — while being wrong only 2.3% of the time in
    # rainfed Kharif. The envelopes were right; the water assumption was not.
    hard = {k: v for k, v in factors.items() if k in HARD_FACTORS}
    hard_limiting = min(hard, key=hard.get) if hard else None
    hard_fails = bool(hard) and hard[hard_limiting] < 0.25

    if irrigated:
        best = total
    else:
        water_relieved = dict(factors)
        for f in WATER_FACTORS:
            if f in water_relieved and rain < req.rain_mm.opt_min:
                water_relieved[f] = 1.0
        best = min(water_relieved.values())

    vetoed = hard_fails or best < 0.25
    needs_water = (not vetoed) and total < 0.25 <= best

    return Suitability(
        crop=crop, season=season, score=round(float(total), 3),
        score_irrigated=round(float(best), 3),
        requires_irrigation=bool(needs_water),
        hard_limiting_factor=hard_limiting if hard_fails else None,
        suitability_class=code, label=label, limiting_factor=limiting,
        factors={k: round(v, 3) for k, v in factors.items()},
        evidence={"effective_water_mm": round(rain, 1),
                  "irrigated": irrigated,
                  "season_rain_mm": round(_season_value(features, "rain", season), 1),
                  "tmean_c": round(tmean, 1),
                  "pH": round(ph, 2), "depth_mm": depth, "drainage_ord": drainage,
                  "saline_pct": round(saline, 2), "lgp_days": lgp},
        vetoed=vetoed,
        reason=_explain(req, limiting, factors, code, vetoed, needs_water,
                        hard_limiting if hard_fails else None),
    )


def _explain(req: CropRequirement, limiting: str, factors: dict, code: str,
             vetoed: bool = False, needs_water: bool = False,
             hard_factor: str | None = None) -> str:
    """Plain-language reason drawn from the limiting factor."""
    phrases = {
        "rain": "seasonal rainfall is outside the crop's range",
        "temp": "season temperature is outside the crop's range",
        "pH": "soil pH is outside the crop's range",
        "depth": "the soil profile is too shallow for this crop's root system",
        "drainage": "soil drainage does not match what this crop needs",
        "salinity": "too large a share of samples is saline for this crop",
        "LGP": "the growing period is too short for this crop's duration",
        "texture": "soil texture is not this crop's preference",
    }
    head = phrases[limiting]
    if needs_water:
        return (f"Viable only with irrigation: {head} "
                f"(rainfed score {factors[limiting]:.2f}). The land itself is "
                f"suitable; the water is not there without irrigation.")
    if vetoed:
        blame = hard_factor or limiting
        return (f"Vetoed: {phrases[blame]} (score {factors[blame]:.2f}). "
                f"{req.notes}").strip()
    if factors[limiting] >= 0.99:
        return "No limiting factor — every agronomic requirement is comfortably met."
    if code == "S1":
        return (f"Well suited; the tightest factor is {limiting} at "
                f"{factors[limiting]:.2f}, which is not constraining.")
    return f"Limited by {limiting}: {head} (score {factors[limiting]:.2f})."


def score_all(
    features: dict,
    season: str,
    crops: list[str] | None = None,
    *,
    irrigated: bool = False,
    include_out_of_season: bool = False,
) -> list[Suitability]:
    """Score every encoded crop for a taluka-season, best first.

    Crops not grown in the requested season are excluded by default: a calendar
    mismatch is a different kind of answer from "this land cannot support it",
    and mixing the two makes the ranked list unreadable.
    """
    crops = crops or covered_crops()
    results = [score(c, season, features, irrigated=irrigated) for c in crops]
    out = [
        r for r in results
        if r.scored and (include_out_of_season or r.limiting_factor != "season")
    ]
    return sorted(out, key=lambda r: -r.score)


def gate(candidates: list[str], season: str, features: dict,
         irrigated: bool = False) -> tuple[list[str], list[Suitability]]:
    """Hard veto: split candidates into survivors and a logged veto list.

    Crops with no envelope pass through unvetoed — the gate never rejects what
    it cannot assess.
    """
    survivors, vetoes = [], []
    for crop in candidates:
        s = score(crop, season, features, irrigated=irrigated)
        if s.scored and s.vetoed:
            vetoes.append(s)
        else:
            survivors.append(crop)
    return survivors, vetoes


def water_limited(features: dict, season: str, min_share: float = 0.5) -> bool:
    """True when this taluka-season can crop, but mostly only with irrigation.

    Since the veto now fires only on physical impossibility, "water limited" is
    no longer "everything is vetoed" — it is "most of what is viable here needs
    water supplied". That is the actionable statement, and it is the clearest
    demonstration of why irrigation coverage is the highest-value dataset to
    add (plan §8, item 3).
    """
    viable = [s for s in score_all(features, season) if not s.vetoed]
    if not viable:
        return False
    return sum(s.requires_irrigation for s in viable) / len(viable) >= min_share
