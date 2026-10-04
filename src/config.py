"""Single source of truth for paths, constants and conventions.

Every module imports from here; nothing hard-codes a path or a magic number.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
FEATURES = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"

for _d in (INTERIM, FEATURES, REPORTS, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)

# ---- raw files -------------------------------------------------------------
F_SOIL_TYPE = RAW / "maharashtra_soil_type_talukas.csv"
F_FERT = RAW / "maharashtra_fertilizer_recommendations.csv"
F_COORDS = RAW / "taluka_coords.json"

#: The soil-photo classifier's metadata, in the app tree beside its checkpoint
#: rather than in the engine's data directory — the app trains and serves it,
#: the engine only reads its confusion matrix to weigh a photograph against the
#: taluka survey (src/rules/soil_fusion.py).
#:
#: This must name the classifier that is **actually serving**, which is the one
#: `backend/models.py:_soil_model()` loads from `ML/models/`. Pointing it at a
#: challenger in `ML/models/soil_v2/` would have fusion weigh photographs by the
#: confusion matrix of a model no request ever runs — the likelihoods would
#: belong to a different classifier than the probabilities. Promotion means
#: copying a winner's files into `ML/models/`, and this path then follows it.
#:
#: Absent is a normal state: with no classifier, fusion abstains and the survey
#: stands.
#:
#: In a container the app tree is not beside the engine, so the deploy copies the
#: serving classifier's metadata into the image and names it here. It must be a
#: copy of exactly what `ML/models/` serves — see `scripts/sync_engine_assets.sh`.
F_SOIL_MODEL_META = Path(
    os.environ.get("AGROSENSE_SOIL_MODEL_META")
    or ROOT.parent / "ML" / "models" / "soil_metadata.json"
)

#: Soil Health Card cycles, oldest first. Coverage grows 347 -> 350 -> 351
#: talukas; see src/data/admin_changes.py for why that is survey coverage and
#: not new administrative units.
SHC_CYCLES = ["2023-24", "2024-25", "2025-26"]
F_SHC_BY_CYCLE = {c: RAW / f"maharashtra_all_parameters_{c}_talukas.csv"
                  for c in SHC_CYCLES}
#: the cycle used wherever a single static soil description is wanted
SHC_REFERENCE_CYCLE = "2025-26"
F_SHC = F_SHC_BY_CYCLE[SHC_REFERENCE_CYCLE]

#: Agricultural weather years (April -> March), oldest first, **discovered from
#: the files themselves** rather than hand-listed: there are 29 of them now and
#: a list that long, maintained by hand, is a place for a year to go missing.
#:
#: The panel used to hold three years, all of them *after* every crop label, so
#: weather could only ever be a climatology descriptor. It now runs 1997-98 to
#: 2025-26, fetched from the same NASA POWER API with the same parameters as the
#: delivered files (scripts/ingest_weather.py checks every one against its own
#: Date column before it is allowed in), so every crop year finally has the
#: weather it was actually grown in.
#:
#: The delivered file named "2022-04-01_to_2023-03-31" was a byte-identical
#: duplicate of the 2025-26 file whose own dates read 2025-26. It is kept in
#: data/raw/_rejected/ and the genuine 2022-23 weather now occupies that name.
def _discover_weather_years() -> dict[str, tuple[Path, str, str]]:
    found: dict[str, tuple[Path, str, str]] = {}
    for path in sorted(RAW.glob("maharashtra_daily_weather_taluka_*.csv")):
        lo, hi = path.stem.rsplit("taluka_", 1)[1].split("_to_")
        found[f"{lo[:4]}-{hi[2:4]}"] = (path, lo, hi)   # 1997-04-01 .. 1998-03-31 -> "1997-98"
    return found


_WEATHER_FILES = _discover_weather_years()
WEATHER_YEARS = list(_WEATHER_FILES)
F_WEATHER_BY_YEAR = {y: v[0] for y, v in _WEATHER_FILES.items()}
F_WEATHER = F_WEATHER_BY_YEAR[WEATHER_YEARS[-1]] if WEATHER_YEARS else None

#: Expected internal date span per weather year, asserted in the tests so a
#: mislabelled file can never enter the pipeline unnoticed again.
WEATHER_YEAR_SPANS = {y: (v[1], v[2]) for y, v in _WEATHER_FILES.items()}

#: APY crop-statistics years, oldest first. Eight years replaces the single
#: year the original plan was sized for: 34 district labels become 272.
APY_YEARS = ["2015-2016", "2016-2017", "2017-2018", "2018-2019",
             "2019-2020", "2020-2021", "2021-2022", "2022-2023"]
F_APY_BY_YEAR = {
    "2015-2016": RAW / "maharashtra_crops_apy_2015-16.csv",
    "2016-2017": RAW / "maharashtra_crops_apy_2016-17.csv",
    "2017-2018": RAW / "maharashtra_crops_apy_2017-18.csv",
    "2018-2019": RAW / "maharashtra_crops_apy_2018-19.csv",
    "2019-2020": RAW / "maharashtra_crops_apy_2019-20.csv",
    "2020-2021": RAW / "maharashtra_crops_apy_2020-21.csv",
    "2021-2022": RAW / "maharashtra_crops_apy_2021-22.csv",
    "2022-2023": RAW / "maharashtra_crops_apy_2025-26.csv",
}
F_APY = F_APY_BY_YEAR["2022-2023"]

#: APY crop year -> the weather year it was actually grown in.
#:
#: This was empty for the whole life of the project, and that emptiness was the
#: single largest limitation in every report: "weather is the wrong year", so no
#: model could learn a dry-year from a wet-year response. With the backfill it
#: is full — each crop year maps to its own April-March weather.
#:
#: Year-matched weather is legitimate for the YIELD model, which explains an
#: outcome after the season happened; skill measured with it is an upper bound
#: on what can be forecast, and is reported as one. It is NOT legitimate for the
#: ranker, which answers before the season: that gets normals computed as-of the
#: query year only (src/features/climatology.py).
APY_WEATHER_MATCH: dict[str, str] = {
    y: f"{y[:4]}-{y[7:9]}" for y in APY_YEARS
    if f"{y[:4]}-{y[7:9]}" in F_WEATHER_BY_YEAR
}

#: Years held out for the temporal (forward-in-time) validation split.
TEMPORAL_HOLDOUT_YEARS = ["2021-2022", "2022-2023"]

SEED = 42

# ---- keys ------------------------------------------------------------------
# Plan §2 fact 4: six taluka names are duplicated across districts
# (Ashti, Kalamb, Karanja, Karjat, Khed, Malegaon). Joining on Taluka alone
# silently corrupts those rows, so (District, Taluka) is THE key everywhere.
KEY = ["District", "Taluka"]
DISTRICT_KEY = "District"

# Plan §2: seven urban talukas appear in soil/weather but have no Soil Health
# Card record. Dropped with an explicit flag rather than imputed.
URBAN_NO_SHC = {
    ("MUMBAI SUBURBAN", "ANDHERI"),
    ("MUMBAI SUBURBAN", "BORIVALI"),
    ("MUMBAI SUBURBAN", "KURLA"),
    ("NAGPUR", "NAGPUR (URBAN)"),
    ("PUNE", "PUNE CITY"),
    ("THANE", "THANE"),
    ("THANE", "ULHASNAGAR"),
}

# ---- seasons (plan §4.3) ---------------------------------------------------
# Month numbers per agro-season used to collapse 365 daily rows per taluka.
SEASON_MONTHS = {
    "Kharif": [6, 7, 8, 9],
    "Rabi": [10, 11, 12, 1],
    "Summer": [2, 3, 4, 5],
}
SEASONS = list(SEASON_MONTHS)
# APY also carries a "Whole Year" season for perennials such as sugarcane.
APY_SEASONS = SEASONS + ["Whole Year"]

# ---- soil-health nutrient bands (plan §6.1) --------------------------------
# National SHC rating bands. The fertiliser table's three archetypes are
# exactly the band midpoints — verified against the raw table.
SOIL_BANDS = {
    "N": {"low_max": 280.0, "high_min": 560.0},
    "P": {"low_max": 10.0, "high_min": 25.0},
    "K": {"low_max": 108.0, "high_min": 280.0},
    "OC": {"low_max": 0.50, "high_min": 0.75},
}
# Verified archetype anchors carried by the fertiliser table itself.
SOIL_ARCHETYPES = {
    "N": {"Low": 200.0, "Medium": 400.0, "High": 700.0},
    "P": {"Low": 6.0, "Medium": 17.0, "High": 40.0},
    "K": {"Low": 80.0, "Medium": 190.0, "High": 350.0},
    "OC": {"Low": 0.3, "Medium": 0.6, "High": 1.0},
}
SOIL_CLASSES = ["Low", "Medium", "High"]

# ---- soil physical mappings (plan §4.2) ------------------------------------
# Available water capacity by texture. The plan's §4.2 table assumes a
# Sandy -> Clayey vocabulary; the data actually carries only Clayey, Loamy and
# two "-skeletal" variants. Skeletal soils hold >35% coarse fragments, so the
# fine-earth AWC is discounted ~40%.
AWC_MM_PER_M = {
    "Sandy": 60.0,
    "Sandy loam": 100.0,
    "Loamy-skeletal": 85.0,
    "Loamy": 140.0,
    "Clayey-skeletal": 110.0,
    "Clayey": 180.0,
}
# NBSS depth classes. "Moderately deep" fills the plan's "Medium" slot.
DEPTH_MM = {
    "Very shallow": 150.0,
    "Shallow": 300.0,
    "Moderately deep": 600.0,
    "Medium": 600.0,
    "Deep": 1000.0,
    "Very deep": 1500.0,
}
DRAINAGE_ORD = {           # ordered scale, poorly -> excessively
    "Poorly drained": 1,
    "Imperfectly drained": 2,
    "Moderately well drained": 3,
    "Well drained": 4,
    "Somewhat excessively drained": 5,
    "Excessively drained": 6,
}
# Soil_pH_Class is a coarse label; "Others" is a genuine unknown, kept as NaN.
PH_CLASS_VALUE = {
    "Strongly acidic": 4.8,
    "Moderately acidic": 5.6,
    "Slightly acidic": 6.4,
    "Neutral": 7.0,
    "Slightly alkaline": 7.8,
}

# ---- agro-climatic thresholds (plan §4.3) ----------------------------------
DRY_DAY_MM = 2.5           # a "dry day" for dry-spell counting
HEAVY_RAIN_MM = 65.0
GDD_BASE_C = 10.0
HEAT_STRESS_C = 35.0
SEVERE_HEAT_C = 40.0
COLD_STRESS_C = 10.0
FUNGAL_RH_PCT = 80.0
FUNGAL_TMAX_RANGE = (25.0, 32.0)
MONSOON_ONSET_WINDOW_MM = 25.0   # 5-day cumulative
MONSOON_ONSET_WINDOW_DAYS = 5

# ---- micronutrient correction layer (plan §6.3) ----------------------------
# Deficiency share (% of samples) above which the correction is triggered.
MICRO_DEFICIENCY_TRIGGER_PCT = 40.0
MICRONUTRIENTS = ["S", "Fe", "Zn", "Cu", "B", "Mn"]

# ---- modelling (plan §5, §7) ----------------------------------------------
N_FOLDS = 6
N_SEEDS = 10               # seed bagging, §7.2
#: seeds averaged in the ranker that is actually SERVED. Evaluation bagged
#: three seeds while serving shipped one, so the model a farmer met was not the
#: model the report described. Both sides now average the same way.
RANKER_SERVED_SEEDS = 5
N_FEATURES_SELECTED = 35   # nested importance selection optimum, §7.2
QUANTILES = [0.1, 0.5, 0.9]
RANKER_RELEVANCE_GRADES = 5
ALPHA_RHO_FLOOR = 0.2      # below this per-crop skill, rules carry the call (§5.4)
CONFORMAL_ALPHAS = [0.1, 0.2]   # 90% and 80% intervals
