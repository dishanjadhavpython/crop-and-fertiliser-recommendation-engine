"""Land-suitability requirement envelopes (plan §5.2).

Knowledge-based, from FAO EcoCrop and ICAR package-of-practices. Needs no
training data, which is why it covers crops with no yield record at all and is
fully explainable — for a farmer-facing system and for a viva, that matters.

Each envelope is a trapezoid: ``abs_min`` and ``abs_max`` bound the crop's
survival range, ``opt_min`` to ``opt_max`` is where it performs.

Per plan §9 week 3, the top 25 crops are encoded first; the schema extends to
all 78 fertiliser-table crops without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Envelope:
    """A trapezoidal requirement on one continuous factor."""
    abs_min: float
    opt_min: float
    opt_max: float
    abs_max: float


@dataclass(frozen=True)
class CropRequirement:
    """The full land-suitability envelope for one crop."""
    name: str
    seasons: tuple[str, ...]
    rain_mm: Envelope                  # over the growing season
    temp_c: Envelope                   # mean temperature over the season
    ph: Envelope
    min_depth_mm: float
    #: drainage ordinal 1 (poorly) - 6 (excessively); see config.DRAINAGE_ORD
    drainage: Envelope
    #: tolerated share of saline samples, %
    max_saline_pct: float
    min_lgp_days: float
    preferred_textures: tuple[str, ...] = ()
    duration_days: int = 120
    notes: str = ""


def _e(a, b, c, d) -> Envelope:
    return Envelope(a, b, c, d)


K, R, S, W = "Kharif", "Rabi", "Summer", "Whole Year"

#: crop name (fertiliser-table spelling) -> requirement envelope
CROP_REQUIREMENTS: dict[str, CropRequirement] = {
    # ---------------------------------------------------------- cereals ----
    "Rice": CropRequirement(
        "Rice", (K, S), rain_mm=_e(400, 900, 2200, 7000), temp_c=_e(15, 22, 32, 40),
        ph=_e(4.5, 5.5, 7.0, 8.5), min_depth_mm=300, drainage=_e(0.5, 1, 3.5, 5.0),
        max_saline_pct=8, min_lgp_days=100, preferred_textures=("Clayey", "Clayey-skeletal"),
        duration_days=125,
        notes="Wants a low-permeability profile that holds standing water; "
              "excessively drained uplands are the classic misrecommendation."),
    "Wheat": CropRequirement(
        "Wheat", (R,), rain_mm=_e(120, 300, 900, 2000), temp_c=_e(5, 15, 25, 33),
        ph=_e(5.0, 6.0, 7.5, 8.5), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=15, min_lgp_days=90, preferred_textures=("Loamy", "Clayey"),
        duration_days=120,
        notes="Rabi crop; needs cool grain-filling temperatures and assured irrigation."),
    "Sorghum": CropRequirement(
        "Sorghum", (K, R), rain_mm=_e(200, 400, 1000, 4000), temp_c=_e(14, 24, 33, 42),
        ph=_e(5.0, 5.5, 8.2, 9.0), min_depth_mm=300, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=25, min_lgp_days=75, preferred_textures=("Clayey", "Loamy"),
        duration_days=110,
        notes="Drought-hardy and salt-tolerant; the safe fallback on shallow black soils."),
    "Pearl Millet": CropRequirement(
        "Pearl Millet", (K, S), rain_mm=_e(120, 300, 800, 2000), temp_c=_e(16, 25, 35, 43),
        ph=_e(5.0, 5.5, 8.5, 9.2), min_depth_mm=200, drainage=_e(2.0, 3.5, 6.0, 6.5),
        max_saline_pct=30, min_lgp_days=60, preferred_textures=("Loamy", "Loamy-skeletal"),
        duration_days=85,
        notes="The most drought-tolerant cereal here; performs on shallow, coarse soils "
              "where nothing else will."),
    "Maize": CropRequirement(
        "Maize", (K, R, S), rain_mm=_e(300, 500, 1300, 5000), temp_c=_e(12, 20, 30, 38),
        ph=_e(5.0, 5.8, 7.5, 8.3), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=12, min_lgp_days=90, preferred_textures=("Loamy", "Clayey"),
        duration_days=110,
        notes="Sensitive to both waterlogging and moisture stress at flowering."),
    "Finger Millet": CropRequirement(
        "Finger Millet", (K,), rain_mm=_e(250, 500, 1300, 3000), temp_c=_e(12, 20, 30, 36),
        ph=_e(4.5, 5.0, 8.2, 8.8), min_depth_mm=250, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=15, min_lgp_days=85, preferred_textures=("Loamy",),
        duration_days=105,
        notes="Hill and lateritic tracts; tolerates acidity better than most cereals."),

    # ----------------------------------------------------------- pulses ----
    "Chickpea": CropRequirement(
        "Chickpea", (R,), rain_mm=_e(120, 250, 750, 1600), temp_c=_e(8, 18, 27, 35),
        ph=_e(5.5, 6.0, 8.2, 9.0), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=18, min_lgp_days=85, preferred_textures=("Clayey", "Loamy"),
        duration_days=105,
        notes="Rabi pulse on residual moisture; waterlogging is fatal."),
    "Pigeon Pea": CropRequirement(
        "Pigeon Pea", (K,), rain_mm=_e(250, 500, 1400, 3000), temp_c=_e(15, 20, 32, 40),
        ph=_e(5.0, 5.5, 7.8, 8.5), min_depth_mm=600, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=10, min_lgp_days=120, preferred_textures=("Loamy", "Clayey"),
        duration_days=180,
        notes="Deep taproot needs profile depth; extremely waterlogging-sensitive."),
    "Mungbean": CropRequirement(
        "Mungbean", (K, S), rain_mm=_e(150, 300, 800, 1600), temp_c=_e(18, 25, 33, 41),
        ph=_e(5.5, 6.2, 7.5, 8.5), min_depth_mm=300, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=8, min_lgp_days=60, preferred_textures=("Loamy",),
        duration_days=70,
        notes="Short duration; fits a gap between two main crops."),
    "Urdbean": CropRequirement(
        "Urdbean", (K, S), rain_mm=_e(200, 350, 900, 1800), temp_c=_e(18, 25, 33, 40),
        ph=_e(5.5, 6.0, 7.8, 8.5), min_depth_mm=300, drainage=_e(1.5, 2.5, 5.5, 6.5),
        max_saline_pct=8, min_lgp_days=65, preferred_textures=("Loamy", "Clayey"),
        duration_days=80,
        notes="Slightly more moisture-tolerant than mungbean."),

    # --------------------------------------------------------- oilseeds ----
    "Groundnut": CropRequirement(
        "Groundnut", (K, S), rain_mm=_e(250, 450, 1200, 2500), temp_c=_e(15, 24, 33, 40),
        ph=_e(5.0, 6.0, 7.5, 8.2), min_depth_mm=450, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=8, min_lgp_days=100, preferred_textures=("Loamy", "Loamy-skeletal"),
        duration_days=115,
        notes="Needs a loose, well-drained profile for pegging; heavy clay is a poor fit."),
    "Soybean": CropRequirement(
        "Soybean", (K,), rain_mm=_e(300, 500, 1300, 2500), temp_c=_e(14, 20, 32, 38),
        ph=_e(5.0, 6.0, 7.5, 8.3), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=8, min_lgp_days=95, preferred_textures=("Clayey", "Loamy"),
        duration_days=100,
        notes="The dominant Kharif oilseed of Vidarbha and Marathwada."),
    "Sunflower": CropRequirement(
        "Sunflower", (R, S, K), rain_mm=_e(150, 350, 900, 2000), temp_c=_e(12, 20, 30, 38),
        ph=_e(5.5, 6.2, 8.0, 8.8), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=20, min_lgp_days=85, preferred_textures=("Loamy", "Clayey"),
        duration_days=95,
        notes="Photoperiod-insensitive, so it fits all three seasons."),
    "Safflower": CropRequirement(
        "Safflower", (R,), rain_mm=_e(120, 220, 650, 1400), temp_c=_e(10, 18, 28, 36),
        ph=_e(5.5, 6.5, 8.5, 9.2), min_depth_mm=600, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=30, min_lgp_days=90, preferred_textures=("Clayey",),
        duration_days=120,
        notes="Deep taproot on residual moisture in deep black soils; very salt-tolerant."),
    "Sesame": CropRequirement(
        "Sesame", (K, R, S), rain_mm=_e(150, 350, 850, 1600), temp_c=_e(18, 25, 33, 40),
        ph=_e(5.5, 6.0, 8.0, 8.7), min_depth_mm=300, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=6, min_lgp_days=75, preferred_textures=("Loamy", "Loamy-skeletal"),
        duration_days=90,
        notes="Extremely waterlogging-sensitive — 48 hours of standing water kills it."),
    "Indian Mustard": CropRequirement(
        "Indian Mustard", (R,), rain_mm=_e(120, 250, 700, 1400), temp_c=_e(7, 15, 27, 34),
        ph=_e(5.5, 6.0, 8.0, 8.8), min_depth_mm=400, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=20, min_lgp_days=85, preferred_textures=("Loamy", "Clayey"),
        duration_days=110,
        notes="Rabi oilseed on residual moisture; tolerates salinity better than "
              "most oilseeds. Appears in the APY panel only for 2016-17..2019-20."),
    "Linseed": CropRequirement(
        "Linseed", (R,), rain_mm=_e(120, 250, 750, 1500), temp_c=_e(8, 15, 26, 33),
        ph=_e(5.0, 6.0, 7.5, 8.3), min_depth_mm=400, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=12, min_lgp_days=90, preferred_textures=("Clayey", "Loamy"),
        duration_days=110, notes="Cool-season Rabi oilseed."),

    # ------------------------------------------------- commercial crops ----
    "Sugarcane": CropRequirement(
        "Sugarcane", (W,), rain_mm=_e(500, 1000, 2600, 5000), temp_c=_e(15, 24, 33, 40),
        ph=_e(5.0, 6.0, 7.7, 8.5), min_depth_mm=900, drainage=_e(1.5, 2.5, 5.0, 6.0),
        max_saline_pct=12, min_lgp_days=150, preferred_textures=("Clayey", "Loamy"),
        duration_days=330,
        notes="A 12-month crop with very high water demand; deep profile essential. "
              "Its Maharashtra footprint is partly co-operative politics, not agronomy."),
    "Tetraploid Cotton": CropRequirement(
        "Tetraploid Cotton", (K,), rain_mm=_e(300, 500, 1300, 2500), temp_c=_e(15, 24, 34, 42),
        ph=_e(5.5, 6.0, 8.2, 9.0), min_depth_mm=600, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=25, min_lgp_days=130, preferred_textures=("Clayey",),
        duration_days=180,
        notes="Deep black cotton soil; heavily irrigation- and market-determined, "
              "which is why the learned model scores below the mean on it (§7.4)."),

    # --------------------------------------------------- horticulture ----
    "Onion": CropRequirement(
        "Onion", (R, K, S), rain_mm=_e(150, 350, 1000, 2000), temp_c=_e(8, 15, 28, 35),
        ph=_e(5.5, 6.0, 7.5, 8.3), min_depth_mm=400, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=6, min_lgp_days=90, preferred_textures=("Loamy",),
        duration_days=130, notes="Nashik belt; bulb rot follows any waterlogging."),
    "Tomato": CropRequirement(
        "Tomato", (R, K, S), rain_mm=_e(200, 400, 1100, 2200), temp_c=_e(10, 20, 29, 36),
        ph=_e(5.0, 6.0, 7.2, 8.0), min_depth_mm=400, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=8, min_lgp_days=90, preferred_textures=("Loamy",),
        duration_days=120, notes="Fruit set fails above ~35 C."),
    "Chilli": CropRequirement(
        "Chilli", (K, R), rain_mm=_e(250, 500, 1300, 2500), temp_c=_e(15, 20, 32, 38),
        ph=_e(5.5, 6.0, 7.5, 8.3), min_depth_mm=400, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=8, min_lgp_days=120, preferred_textures=("Loamy", "Clayey"),
        duration_days=160),
    "Potato": CropRequirement(
        "Potato", (R,), rain_mm=_e(150, 350, 900, 1800), temp_c=_e(7, 15, 24, 30),
        ph=_e(4.5, 5.0, 6.8, 7.8), min_depth_mm=400, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=5, min_lgp_days=85, preferred_textures=("Loamy", "Loamy-skeletal"),
        duration_days=100, notes="Tuber bulking stops above 25 C."),
    "Turmeric": CropRequirement(
        "Turmeric", (W,), rain_mm=_e(600, 1000, 2600, 4000), temp_c=_e(15, 22, 32, 38),
        ph=_e(4.5, 5.5, 7.5, 8.2), min_depth_mm=450, drainage=_e(2.0, 3.0, 5.5, 6.5),
        max_saline_pct=6, min_lgp_days=180, preferred_textures=("Loamy", "Clayey"),
        duration_days=250, notes="Long duration, high moisture, but rhizomes rot if waterlogged."),
    "Banana": CropRequirement(
        "Banana", (W,), rain_mm=_e(600, 1100, 2700, 4000), temp_c=_e(15, 24, 33, 40),
        ph=_e(5.0, 6.0, 7.5, 8.2), min_depth_mm=750, drainage=_e(2.0, 3.5, 5.5, 6.5),
        max_saline_pct=5, min_lgp_days=200, preferred_textures=("Loamy", "Clayey"),
        duration_days=330, notes="Jalgaon belt; needs deep soil and continuous water."),
    "Grapes": CropRequirement(
        "Grapes", (W,), rain_mm=_e(150, 350, 1000, 2500), temp_c=_e(10, 20, 33, 42),
        ph=_e(5.5, 6.5, 8.2, 9.0), min_depth_mm=750, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=15, min_lgp_days=120, preferred_textures=("Loamy", "Clayey"),
        duration_days=330,
        notes="Rain at ripening causes berry cracking, so a dry finish matters more "
              "than the annual total. High diurnal range favours sugar accumulation."),
    "Pomegranate": CropRequirement(
        "Pomegranate", (W,), rain_mm=_e(120, 250, 900, 2000), temp_c=_e(10, 22, 36, 45),
        ph=_e(5.5, 6.5, 8.5, 9.2), min_depth_mm=600, drainage=_e(2.5, 3.5, 6.0, 6.5),
        max_saline_pct=25, min_lgp_days=100, preferred_textures=("Loamy", "Loamy-skeletal"),
        duration_days=330,
        notes="Semi-arid Solapur/Sangli belt; tolerates drought and salinity, "
              "hates humidity (bacterial blight)."),
}

#: APY aggregates with no meaningful single envelope. The gate abstains rather
#: than inventing requirements for a bucket of unlike crops.
NO_ENVELOPE: frozenset[str] = frozenset({
    "Other Cereals", "Other Kharif pulses", "Other Rabi pulses",
    "Other Summer Pulses", "other oilseeds", "Niger seed", "Tobacco",
})


def get(crop: str) -> CropRequirement | None:
    """Case-insensitive envelope lookup."""
    if crop in CROP_REQUIREMENTS:
        return CROP_REQUIREMENTS[crop]
    fold = crop.casefold()
    for name, req in CROP_REQUIREMENTS.items():
        if name.casefold() == fold:
            return req
    return None


def covered_crops() -> list[str]:
    return sorted(CROP_REQUIREMENTS)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
# Every envelope above is a set of numbers, and a number without a source is
# indefensible in review. This registry names, per crop, the binomial (the
# unambiguous key for looking the crop up) and the authorities the envelope is
# drawn from.
#
# Read the status honestly: these envelopes are *encoded from* standard
# agronomic references — FAO EcoCrop's ecological tolerance ranges and the
# relevant ICAR institute's Package of Practices — and then narrowed to
# Maharashtra practice using the observed cropping pattern. They have NOT been
# checked line by line against the primary records. ``VERIFIED_AGAINST_SOURCE``
# is therefore empty, and it should be populated one crop at a time as each is
# actually checked. An empty set is a true statement; a full one would not be.
#
# The task this makes tractable: `python -m src.rules.crop_requirements` prints
# the verification worksheet, one row per crop per factor.

CROP_SOURCES: dict[str, dict[str, str]] = {
    "Rice": {"species": "Oryza sativa", "source": "FAO EcoCrop; ICAR-IIRR Hyderabad PoP"},
    "Wheat": {"species": "Triticum aestivum", "source": "FAO EcoCrop; ICAR-IIWBR Karnal PoP"},
    "Sorghum": {"species": "Sorghum bicolor", "source": "FAO EcoCrop; ICAR-IIMR Hyderabad PoP"},
    "Pearl Millet": {"species": "Pennisetum glaucum", "source": "FAO EcoCrop; ICRISAT agro-ecology"},
    "Maize": {"species": "Zea mays", "source": "FAO EcoCrop; ICAR-IIMR Ludhiana PoP"},
    "Finger Millet": {"species": "Eleusine coracana", "source": "FAO EcoCrop; ICAR-IIMR PoP"},
    "Chickpea": {"species": "Cicer arietinum", "source": "FAO EcoCrop; ICAR-IIPR Kanpur PoP"},
    "Pigeon Pea": {"species": "Cajanus cajan", "source": "FAO EcoCrop; ICRISAT; ICAR-IIPR"},
    "Mungbean": {"species": "Vigna radiata", "source": "FAO EcoCrop; ICAR-IIPR Kanpur PoP"},
    "Urdbean": {"species": "Vigna mungo", "source": "FAO EcoCrop; ICAR-IIPR Kanpur PoP"},
    "Groundnut": {"species": "Arachis hypogaea", "source": "FAO EcoCrop; ICAR-DGR Junagadh PoP"},
    "Soybean": {"species": "Glycine max", "source": "FAO EcoCrop; ICAR-IISR Indore PoP"},
    "Sunflower": {"species": "Helianthus annuus", "source": "FAO EcoCrop; ICAR-IIOR Hyderabad PoP"},
    "Safflower": {"species": "Carthamus tinctorius", "source": "FAO EcoCrop; ICAR-IIOR Hyderabad PoP"},
    "Sesame": {"species": "Sesamum indicum", "source": "FAO EcoCrop; ICAR-IIOR Hyderabad PoP"},
    "Linseed": {"species": "Linum usitatissimum", "source": "FAO EcoCrop; ICAR Project Coordinator (Linseed)"},
    "Sugarcane": {"species": "Saccharum officinarum", "source": "FAO EcoCrop; Vasantdada Sugar Institute Pune"},
    "Tetraploid Cotton": {"species": "Gossypium hirsutum", "source": "FAO EcoCrop; ICAR-CICR Nagpur PoP"},
    "Indian Mustard": {"species": "Brassica juncea", "source": "FAO EcoCrop; ICAR-DRMR Bharatpur PoP"},
    "Onion": {"species": "Allium cepa", "source": "FAO EcoCrop; ICAR-DOGR Rajgurunagar PoP"},
    "Tomato": {"species": "Solanum lycopersicum", "source": "FAO EcoCrop; ICAR-IIHR Bengaluru PoP"},
    "Chilli": {"species": "Capsicum annuum", "source": "FAO EcoCrop; ICAR-IIHR Bengaluru PoP"},
    "Potato": {"species": "Solanum tuberosum", "source": "FAO EcoCrop; ICAR-CPRI Shimla PoP"},
    "Turmeric": {"species": "Curcuma longa", "source": "FAO EcoCrop; ICAR-IISR Kozhikode PoP"},
    "Banana": {"species": "Musa acuminata", "source": "FAO EcoCrop; ICAR-NRCB Tiruchirappalli PoP"},
    "Grapes": {"species": "Vitis vinifera", "source": "FAO EcoCrop; ICAR-NRCG Pune PoP"},
    "Pomegranate": {"species": "Punica granatum", "source": "FAO EcoCrop; ICAR-NRCP Solapur PoP"},
}

#: Crops whose numbers have been checked line by line against the primary
#: record. Deliberately empty — populate it as each crop is actually verified.
VERIFIED_AGAINST_SOURCE: frozenset[str] = frozenset()


def provenance(crop: str) -> dict:
    """Species, source authority and verification status for one crop."""
    req = get(crop)
    meta = CROP_SOURCES.get(req.name if req else crop, {})
    return {
        "crop": crop,
        "species": meta.get("species"),
        "source": meta.get("source"),
        "verified_against_source": (req.name if req else crop) in VERIFIED_AGAINST_SOURCE,
        "has_envelope": req is not None,
    }


def verification_worksheet() -> "pd.DataFrame":
    """One row per crop per factor — the checklist for verifying the envelopes."""
    import pandas as pd

    rows = []
    for name, req in sorted(CROP_REQUIREMENTS.items()):
        meta = CROP_SOURCES.get(name, {})
        for factor, env in (("rainfall_mm", req.rain_mm), ("temperature_C", req.temp_c),
                            ("pH", req.ph), ("drainage_ord", req.drainage)):
            rows.append({
                "crop": name, "species": meta.get("species", ""),
                "factor": factor,
                "abs_min": env.abs_min, "opt_min": env.opt_min,
                "opt_max": env.opt_max, "abs_max": env.abs_max,
                "source": meta.get("source", ""),
                "verified": name in VERIFIED_AGAINST_SOURCE,
            })
        rows.append({"crop": name, "species": meta.get("species", ""),
                     "factor": "min_depth_mm", "abs_min": req.min_depth_mm,
                     "opt_min": "", "opt_max": "", "abs_max": "",
                     "source": meta.get("source", ""),
                     "verified": name in VERIFIED_AGAINST_SOURCE})
    return pd.DataFrame(rows)


def coverage_gaps() -> list[str]:
    """Fertiliser-table crops with no envelope — the S2 gate cannot assess these."""
    from src.data.load import load_fertiliser
    return sorted(set(load_fertiliser()["Crop"].unique()) - set(CROP_REQUIREMENTS))


if __name__ == "__main__":
    from src import config
    ws = verification_worksheet()
    out = config.REPORTS / "envelope_verification_worksheet.csv"
    ws.to_csv(out, index=False)
    print(f"{len(CROP_REQUIREMENTS)} envelopes, {len(ws)} factor rows -> {out}")
    print(f"verified against primary source: {len(VERIFIED_AGAINST_SOURCE)}")
    print(f"fertiliser crops with NO envelope: {len(coverage_gaps())}")
