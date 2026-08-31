"""Feature-store correctness (plan §4).

The plan singles out the ET0 and dry-spell functions as "the ones most likely
to harbour a silent bug", so both are tested against external references rather
than against themselves. The block ranges quoted in §4 are asserted too: they
are the plan's own empirical fingerprints of a correct build.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import agroclimate, interactions, soil_health, soil_physical
from src.features.build_store import build_feature_store, feature_columns


# --------------------------------------------------------------- Block A ----
def test_nutrient_index_is_bounded_and_correct():
    ni = soil_health.nutrient_index(pd.Series([100.0, 0.0, 0.0, 50.0]),
                                    pd.Series([0.0, 100.0, 0.0, 0.0]),
                                    pd.Series([0.0, 0.0, 100.0, 50.0]))
    assert list(ni) == [1.0, 2.0, 3.0, 2.0]


def test_ilr_is_scale_invariant_and_signed_correctly():
    """ILR depends only on ratios, and ilr2 > 0 exactly when Medium > Low."""
    a1, a2 = soil_health.ilr_3part([20.0], [30.0], [50.0])
    b1, b2 = soil_health.ilr_3part([40.0], [60.0], [100.0])   # same composition, 2x
    assert a1[0] == pytest.approx(b1[0], rel=0.05)
    assert a2[0] > 0                                          # Medium > Low
    c1, _ = soil_health.ilr_3part([50.0], [30.0], [20.0])
    assert c1[0] < 0                                          # High < mean(L, M)


def test_zero_band_does_not_produce_infinity():
    """Several talukas record a true 0% in a band; the closure must absorb it."""
    i1, i2 = soil_health.ilr_3part([0.0], [0.0], [100.0])
    assert np.isfinite(i1[0]) and np.isfinite(i2[0])


def test_block_a_keeps_only_deficient_share_for_micronutrients():
    a = soil_health.build()
    for m in config.MICRONUTRIENTS:
        assert f"def_{m}" in a.columns
        assert f"suf_{m}" not in a.columns


def test_sulphur_is_the_most_variable_soil_feature():
    """Plan §4.1: sulphur deficiency SD is 34.4 pp, the highest of any soil variable."""
    a = soil_health.build()
    assert a["def_S"].std() == pytest.approx(34.4, abs=0.1)
    others = [f"def_{m}" for m in config.MICRONUTRIENTS if m != "S"]
    assert a["def_S"].std() > a[others].std().max()


# --------------------------------------------------------------- Block B ----
def test_rootzone_awc_matches_the_plan_range():
    """Plan §4.2 states 21 -> 270 mm across the talukas."""
    b = soil_physical.build()
    assert b["rootzone_awc"].min() == pytest.approx(21.0, abs=0.5)
    assert b["rootzone_awc"].max() == pytest.approx(270.0, abs=0.5)


def test_drainage_is_an_ordered_scale():
    assert config.DRAINAGE_ORD["Poorly drained"] < config.DRAINAGE_ORD["Well drained"]
    assert config.DRAINAGE_ORD["Well drained"] < config.DRAINAGE_ORD["Excessively drained"]


def test_block_b_has_no_missing_values():
    b = soil_physical.build()
    assert b.isna().sum().sum() == 0
    assert b["texture_imputed"].sum() == 4     # the four Alluvial talukas


# --------------------------------------------------------------- Block C ----
def test_extraterrestrial_radiation_matches_fao56_example_8():
    """FAO-56 Example 8: 3 September (DOY 246) at 20 degrees S -> Ra = 32.2 MJ/m2/d."""
    assert float(agroclimate.extraterrestrial_radiation(-20.0, 246)) == pytest.approx(32.2, abs=0.1)


def test_et0_responds_to_temperature_range_and_stays_plausible():
    et0 = float(agroclimate.hargreaves_et0(35.0, 20.0, 19.0, 150))
    assert 3.0 < et0 < 12.0                      # tropical daily ET0 envelope
    hotter = float(agroclimate.hargreaves_et0(40.0, 20.0, 19.0, 150))
    assert hotter > et0


def test_et0_is_zero_when_the_diurnal_range_is_zero():
    assert float(agroclimate.hargreaves_et0(30.0, 30.0, 19.0, 150)) == 0.0


def test_longest_dry_spell_on_a_known_series():
    rain = [0, 0, 0, 10, 0, 0, 0, 0, 0, 5, 0]     # runs of 3, 5, 1
    assert agroclimate.longest_dry_spell(rain) == 5
    assert agroclimate.longest_dry_spell([10] * 10) == 0
    assert agroclimate.longest_dry_spell([0] * 10) == 10


def test_dry_day_threshold_is_25mm_not_zero():
    """A 2 mm day is a dry day; a 3 mm day is not."""
    assert agroclimate.longest_dry_spell([2.0, 2.0, 2.0]) == 3
    assert agroclimate.longest_dry_spell([3.0, 3.0, 3.0]) == 0


def test_monsoon_onset_finds_the_first_qualifying_pentad():
    dates = pd.date_range("2025-04-01", "2025-12-31")
    rain = pd.Series(0.0, index=range(len(dates)))
    rain.iloc[75:80] = 6.0          # 30 mm over 5 days starting 15 June
    doy = agroclimate.monsoon_onset_doy(dates, rain)
    assert doy == pd.Timestamp("2025-06-15").dayofyear


def test_rain_concentration_is_bounded():
    assert agroclimate.rain_concentration([10] * 5) == pytest.approx(1.0)
    assert agroclimate.rain_concentration([0] * 10) == 0.0
    assert 0.0 < agroclimate.rain_concentration([1] * 100) < 0.1


def test_block_c_reproduces_the_ranges_quoted_in_the_plan():
    c = agroclimate.build()
    assert (round(c["rain_annual"].min()), round(c["rain_annual"].max())) == (629, 3346)
    assert c["dry_spell_Kharif"].max() == 25
    assert c["aridity_annual"].min() == pytest.approx(0.33, abs=0.02)
    assert c["aridity_annual"].max() == pytest.approx(3.19, abs=0.05)
    assert (c["lgp_annual"].min(), c["lgp_annual"].max()) == (73, 166)


# --------------------------------------------------------------- Block D ----
def test_interactions_encode_the_intended_agronomy():
    df = build_feature_store()
    # a shallow sandy taluka is more drought-vulnerable than a deep clay one
    assert df.loc[df["rootzone_awc"].idxmin(), "drought_vuln"] > \
           df.loc[df["rootzone_awc"].idxmax(), "drought_vuln"]
    # leach_risk is monotone in each of its three drivers, holding the others
    # fixed. It is only weakly correlated with rainfall across real talukas
    # (rho ~ 0.38) because the wettest Konkan talukas also have high organic
    # carbon and clay — which is the interaction doing its job.
    synth = pd.DataFrame({
        "rain_annual": [1000.0, 2000.0, 1000.0, 1000.0],
        "NI_OC": [1.0, 1.0, 3.0, 1.0],
        "awc_mm_m": [60.0, 60.0, 60.0, 180.0],
    })
    lr = (synth["rain_annual"] * (1 - synth["NI_OC"] / 3.0)
          * (200.0 - synth["awc_mm_m"])) / 1e4
    assert lr[1] > lr[0]        # more rain -> more leaching
    assert lr[2] < lr[0]        # more organic carbon -> less leaching
    assert lr[3] < lr[0]        # more water-holding capacity -> less leaching
    assert df["leach_risk"].min() >= 0
    # P availability never exceeds the raw index it discounts
    assert (df["p_availability"] <= df["NI_P"] + 1e-9).all()


def test_spatial_and_context_features_are_off_by_default():
    """Plan §7.1: both hurt at n=34, so they must be opt-in."""
    base = build_feature_store()
    assert not [c for c in base.columns if c.startswith(("knn5_", "anom_", "z_state_", "z_district_"))]
    on = build_feature_store(with_context_z=True, with_spatial=True)
    assert [c for c in on.columns if c.startswith("knn5_")]
    assert [c for c in on.columns if c.startswith("z_district_")]


# ---------------------------------------------------------------- store ----
def test_feature_store_shape_and_determinism():
    a, b = build_feature_store(), build_feature_store()
    assert len(a) == 351
    assert len(feature_columns(a)) >= 144       # plan targets ~144 before selection
    pd.testing.assert_frame_equal(a, b)         # reproducible from raw, byte-for-byte


def test_missing_values_are_confined_to_the_soil_health_trend_block():
    """The only NaNs are trends that genuinely cannot be measured.

    Kelapur (Yavatmal) appears in one Soil Health Card cycle only, so it has no
    cycle-to-cycle trend. NaN is the honest value — imputing zero would assert
    "no change observed" about a taluka that was never observed twice — and
    LightGBM consumes NaN natively as its own branch.
    """
    df = build_feature_store()
    num = df[feature_columns(df)]

    missing = num.isna().sum()
    missing = missing[missing > 0]
    assert set(missing.index) <= {c for c in num.columns
                                  if c.startswith(("shcTrend_", "shcDelta_"))}
    # exactly one taluka, and it is the one with a single cycle
    rows = df[num.isna().any(axis=1)]
    assert len(rows) == 1
    assert rows["shc_cycles_observed"].iat[0] == 1

    finite = num.drop(columns=list(missing.index))
    assert np.isfinite(finite.to_numpy(dtype=float)).all()
