"""The day-one fertiliser test (plan §10).

    "For a sample of 1,000 keys, assert the engine reproduces the government
     table byte-for-byte before any adjustment layer is applied. If that ever
     breaks, your recommendations are wrong in a way no metric will catch."

Also re-verifies the claim the whole design rests on: the table is a
deterministic function, so a model here could only introduce error.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.load import load_fertiliser
from src.rules import fertiliser as fz
from src.rules.cost_optimiser import compare_with_table, cost_of, optimise
from src.rules.soil_class import SoilTest, classify, classify_nutrient

RNG = np.random.default_rng(config.SEED)


def test_the_table_is_a_deterministic_function():
    """0 ambiguous keys across all clean rows — the premise for a lookup, not a model."""
    clean = load_fertiliser()
    key = ["District", "Crop", "Crop_Variety", "Crop_Irrigation", "Crop_Season",
           "Soil_Class", "Category", "Option", "Fertilizer"]
    counts = clean.groupby(key, dropna=False)["Quantity"].nunique()
    assert (counts > 1).sum() == 0


def test_engine_reproduces_1000_table_keys_exactly():
    """Layer 2 must be the table verbatim, before interpolation or corrections."""
    t = fz._table()
    keys = (t[["District", "Crop", "Crop_Variety", "Crop_Irrigation",
               "Crop_Season", "Soil_Class", "Option"]]
            .drop_duplicates()
            .reset_index(drop=True))
    sample = keys.iloc[RNG.choice(len(keys), size=1000, replace=False)]

    for row in sample.itertuples(index=False):
        got = fz.lookup(
            row.District, row.Crop, row.Soil_Class,
            variety=row.Crop_Variety,
            irrigation=None if pd.isna(row.Crop_Irrigation) else row.Crop_Irrigation,
            season=None if pd.isna(row.Crop_Season) else row.Crop_Season,
            option=int(row.Option),
        )
        assert got is not None, row

        expect = t[
            (t["District"] == row.District)
            & (t["Crop"] == row.Crop)
            & (t["Crop_Variety"] == row.Crop_Variety)
            & (t["Crop_Irrigation"].isna() if pd.isna(row.Crop_Irrigation)
               else t["Crop_Irrigation"] == row.Crop_Irrigation)
            & (t["Crop_Season"].isna() if pd.isna(row.Crop_Season)
               else t["Crop_Season"] == row.Crop_Season)
            & (t["Soil_Class"] == row.Soil_Class)
            & (t["Option"] == row.Option)
        ]
        assert got["products"] == dict(zip(expect["Fertilizer"], expect["Quantity"].astype(float)))
        assert got["target"]["N"] == pytest.approx(float(expect["N_kg_per_ha"].iloc[0]))


def test_recommend_without_interpolation_is_the_table_verbatim():
    features = _akole_features()
    rec = fz.recommend("AHILYANAGAR", "Sorghum", features,
                       soil_test=SoilTest(400, 17, 190, 0.6),
                       interpolate=False, irrigation="Rainfed", season="Rabi")
    direct = fz.lookup("AHILYANAGAR", "Sorghum", "Medium",
                       irrigation="Rainfed", season="Rabi")
    assert rec.exact_table["products"] == direct["products"]
    assert rec.interpolated_target == direct["target"]


# ------------------------------------------------------ layer 1: classes ----
def test_archetypes_are_the_shc_band_midpoints():
    """Plan §6.1 — verified against the raw table, not assumed."""
    t = fz._table()
    for cls in config.SOIL_CLASSES:
        sub = t[t["Soil_Class"] == cls]
        assert sub["Soil_N_kg_per_ha"].unique().tolist() == [config.SOIL_ARCHETYPES["N"][cls]]
        assert sub["Soil_P_kg_per_ha"].unique().tolist() == [config.SOIL_ARCHETYPES["P"][cls]]
        assert sub["Soil_K_kg_per_ha"].unique().tolist() == [config.SOIL_ARCHETYPES["K"][cls]]
        assert sub["Soil_OC_pct"].unique().tolist() == [config.SOIL_ARCHETYPES["OC"][cls]]


def test_soil_class_boundaries_follow_the_shc_bands():
    assert classify_nutrient("N", 279.9) == "Low"
    assert classify_nutrient("N", 280.0) == "Medium"
    assert classify_nutrient("N", 560.1) == "High"
    assert classify_nutrient("OC", 0.49) == "Low"
    assert classify_nutrient("OC", 0.80) == "High"
    for cls, comp_values in (
        ("Low", (200, 6, 80, 0.3)),
        ("Medium", (400, 17, 190, 0.6)),
        ("High", (700, 40, 350, 1.0)),
    ):
        assert classify(SoilTest(*comp_values)) == cls


# ---------------------------------------------- layer 3: interpolation ----
def test_interpolation_reproduces_the_anchors_exactly():
    targets = fz.crop_targets("AHILYANAGAR", "Sorghum", irrigation="Rainfed", season="Rabi")
    for cls, test in (
        ("Low", SoilTest(200, 6, 80, 0.3)),
        ("Medium", SoilTest(400, 17, 190, 0.6)),
        ("High", SoilTest(700, 40, 350, 1.0)),
    ):
        got = fz.interpolate_target(targets, test)
        for nutrient in ("N", "P2O5", "K2O"):
            assert got[nutrient] == pytest.approx(targets[cls][nutrient])


def test_interpolation_removes_the_bucketing_cliff():
    """A farmer at 279 vs 281 kg N/ha must not fall off a cliff (plan §6.2)."""
    targets = fz.crop_targets("AHILYANAGAR", "Sorghum", irrigation="Rainfed", season="Rabi")
    below = fz.interpolate_target(targets, SoilTest(279, 17, 190, 0.6))["N"]
    above = fz.interpolate_target(targets, SoilTest(281, 17, 190, 0.6))["N"]
    assert abs(above - below) < 0.5
    # ... whereas hard bucketing jumps by the full Low->Medium gap
    assert abs(targets["Low"]["N"] - targets["Medium"]["N"]) > 5


def test_interpolation_is_monotone_and_clamped():
    targets = fz.crop_targets("AHILYANAGAR", "Sorghum", irrigation="Rainfed", season="Rabi")
    doses = [fz.interpolate_target(targets, SoilTest(n, 17, 190, 0.6))["N"]
             for n in (150, 200, 300, 400, 550, 700, 900)]
    assert doses == sorted(doses, reverse=True)          # more soil N -> less applied N
    assert doses[0] == pytest.approx(doses[1])           # clamped below the Low anchor
    assert doses[-1] == pytest.approx(doses[-2])         # clamped above the High anchor


# ------------------------------------------- layer 4: micronutrients ----
def test_micronutrient_layer_uses_components_the_table_ignores():
    plan = fz.micronutrient_plan(_akole_features())
    flagged = {m["component"] for m in plan}
    assert flagged <= set(config.MICRONUTRIENTS)
    # none of the six appear anywhere in the government table's own columns
    assert not (flagged & {"N", "P", "K", "OC"})


def test_micronutrient_trigger_respects_the_threshold():
    base = {f"def_{m}": 0.0 for m in config.MICRONUTRIENTS}
    assert fz.micronutrient_plan(base) == []
    assert fz.micronutrient_plan({**base, "def_Zn": 39.0}) == []
    got = fz.micronutrient_plan({**base, "def_Zn": 41.0})
    assert [m["component"] for m in got] == ["Zn"]
    assert fz.micronutrient_plan({**base, "def_Zn": 80.0})[0]["priority"] == "high"


def test_sulphur_swap_conserves_phosphorus():
    products = {"DAP": 100.0, "Urea": 200.0, "MOP": 50.0}
    micro = [{"component": "S", "deficient_pct": 90.0}]
    swap = fz.sulphur_substitution(products, micro)
    p_from_dap = 100.0 * fz.PRODUCT_ANALYSIS["DAP"]["P2O5"]
    p_from_ssp = swap["ssp_kg_ha"] * fz.PRODUCT_ANALYSIS["SSP"]["P2O5"]
    assert p_from_ssp == pytest.approx(p_from_dap, rel=1e-6)
    assert swap["sulphur_supplied_kg_ha"] > 20        # clears the 20-40 kg S/ha correction


def test_no_sulphur_swap_when_sulphur_is_sufficient():
    assert fz.sulphur_substitution({"DAP": 100.0}, []) is None


# --------------------------------------------------- layer 5: schedule ----
def test_split_schedule_scales_with_leaching_risk():
    low = fz.split_schedule(100.0, 5.0, 0.10)
    mid = fz.split_schedule(100.0, 20.0, 0.70)
    high = fz.split_schedule(100.0, 60.0, 0.95)
    assert (low["n_splits"], mid["n_splits"], high["n_splits"]) == (2, 3, 4)
    for sched in (low, mid, high):
        total = sum(s["n_kg_ha"] for s in sched["schedule"])
        assert total == pytest.approx(100.0, abs=0.5)   # splits conserve the dose
        assert sched["phosphorus_potassium"] == "full dose as basal"


# ------------------------------------------------------- layer 6: cost ----
def test_lp_meets_the_nutrient_target():
    got = optimise({"N": 100.0, "P2O5": 50.0, "K2O": 50.0})
    assert got["feasible"]
    for nutrient, want in (("N", 100.0), ("P2O5", 50.0), ("K2O", 50.0)):
        assert got["nutrients_supplied"][nutrient] >= want - 0.5


def test_lp_never_costs_more_than_the_published_options():
    opts = {o: fz.lookup("AHILYANAGAR", "Sorghum", "Medium",
                         irrigation="Rainfed", season="Rabi", option=o)["products"]
            for o in (1, 2)}
    target = fz.lookup("AHILYANAGAR", "Sorghum", "Medium",
                       irrigation="Rainfed", season="Rabi")["target"]
    cmp = compare_with_table(opts, target)
    assert cmp["saving_inr_per_ha"] >= -0.5     # LP is optimal, so never worse


def test_lp_respects_local_unavailability():
    got = optimise({"N": 100.0, "P2O5": 50.0, "K2O": 50.0}, available=["Urea", "SSP", "MOP"])
    assert got["feasible"] and "DAP" not in got["products_kg_ha"]


def test_lp_can_be_forced_to_supply_sulphur():
    plain = optimise({"N": 100.0, "P2O5": 50.0, "K2O": 50.0})
    with_s = optimise({"N": 100.0, "P2O5": 50.0, "K2O": 50.0}, require_sulphur_kg=25.0)
    assert with_s["feasible"]
    assert with_s["nutrients_supplied"]["S"] >= 24.5
    assert plain["nutrients_supplied"]["S"] < with_s["nutrients_supplied"]["S"]


def _akole_features() -> dict:
    df = pd.read_parquet(config.FEATURES / "taluka_features.parquet")
    return df[(df["District"] == "AHILYANAGAR") & (df["Taluka"] == "AKOLE")].iloc[0].to_dict()
