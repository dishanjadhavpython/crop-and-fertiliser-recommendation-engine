"""Single source of truth for paths, constants and conventions.

Every module imports from here; nothing hard-codes a path or a magic number.
"""
from __future__ import annotations

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

#: Soil Health Card cycles, oldest first. Coverage grows 347 -> 350 -> 351
#: talukas; see src/data/admin_changes.py for why that is survey coverage and
#: not new administrative units.
SHC_CYCLES = ["2023-24", "2024-25", "2025-26"]
F_SHC_BY_CYCLE = {c: RAW / f"maharashtra_all_parameters_{c}_talukas.csv"
                  for c in SHC_CYCLES}
#: the cycle used wherever a single static soil description is wanted
SHC_REFERENCE_CYCLE = "2025-26"
F_SHC = F_SHC_BY_CYCLE[SHC_REFERENCE_CYCLE]

#: Agricultural weather years (April -> March), oldest first. 2023-24 is a
#: leap year and carries 366 days.
#:
#: The delivered file named "2022-04-01_to_2023-03-31" is EXCLUDED: it is a
#: byte-identical duplicate of the 2025-26 file (same MD5) and its own Date
#: column reads 2025-04-01..2026-03-31. Including it would double-count one
#: year and, worse, manufacture a false claim that weather is contemporaneous
#: with the 2022-23 crop labels. Three genuine years remain.
WEATHER_YEARS = ["2023-24", "2024-25", "2025-26"]
F_WEATHER_BY_YEAR = {
    "2023-24": RAW / "maharashtra_daily_weather_taluka_2023-04-01_to_2024-03-31.csv",
    "2024-25": RAW / "maharashtra_daily_weather_taluka_2024-04-01_to_2025-03-31.csv",
    "2025-26": RAW / "maharashtra_daily_weather_taluka_2025-04-01_to_2026-03-31.csv",
}
F_WEATHER = F_WEATHER_BY_YEAR["2025-26"]

#: Expected internal date span per weather year, asserted in the tests so a
#: mislabelled file can never enter the pipeline unnoticed again.
WEATHER_YEAR_SPANS = {
    "2023-24": ("2023-04-01", "2024-03-31"),
    "2024-25": ("2024-04-01", "2025-03-31"),
    "2025-26": ("2025-04-01", "2026-03-31"),
}

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

#: APY years with contemporaneous daily weather: NONE.
#: The crop panel runs 2015-16..2022-23; the genuine weather years run
#: 2023-24..2025-26. They do not overlap, so weather remains strictly a
#: *climatology descriptor* — "what this place is typically like" — exactly as
#: the original plan required. Three years is still not a 30-year normal, but
#: unlike one year it does support inter-annual variability features.
APY_WEATHER_MATCH: dict[str, str] = {}

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
N_FEATURES_SELECTED = 35   # nested importance selection optimum, §7.2
QUANTILES = [0.1, 0.5, 0.9]
RANKER_RELEVANCE_GRADES = 5
ALPHA_RHO_FLOOR = 0.2      # below this per-crop skill, rules carry the call (§5.4)
CONFORMAL_ALPHAS = [0.1, 0.2]   # 90% and 80% intervals
