"""S2 agronomic gate (plan §5.2).

The gate's job is to veto the class of error a learned ranker trained on 34
districts will make — "rice on a shallow, excessively-drained upland with
630 mm of rain". These tests assert it catches exactly that, that it combines
factors by Liebig's minimum rather than by averaging, and that it produces the
regionally correct answer for talukas whose cropping pattern is well known.
"""
import pandas as pd
import pytest

from src import config
from src.ontology.crop_map import CROP_ONTOLOGY, to_fertiliser_crop
from src.rules import crop_requirements as cr
from src.rules.suitability import (
    classify_score,
    effective_water,
    gate,
    score,
    score_all,
    trapezoid,
    water_limited,
)

FEATURES = pd.read_parquet(config.FEATURES / "taluka_features.parquet")


def taluka(district: str, name: str) -> dict:
    row = FEATURES[(FEATURES["District"] == district) & (FEATURES["Taluka"] == name)]
    assert not row.empty, f"{district}/{name} not in the feature store"
    return row.iloc[0].to_dict()


# ------------------------------------------------------------- mechanics ----
def test_trapezoid_shape():
    env = cr.Envelope(100, 200, 400, 600)
    assert trapezoid(50, env) == 0.0            # below absolute minimum
    assert trapezoid(150, env) == pytest.approx(0.5)
    assert trapezoid(300, env) == 1.0           # inside the optimum
    assert trapezoid(500, env) == pytest.approx(0.5)
    assert trapezoid(700, env) == 0.0           # above absolute maximum


def test_unknown_factor_does_not_penalise():
    assert trapezoid(float("nan"), cr.Envelope(1, 2, 3, 4)) == 1.0


def test_liebig_minimum_not_average():
    """Excellent rainfall must not mask a fatal pH problem."""
    f = taluka("KOLHAPUR", "KARVIR")
    s = score("Sorghum", "Kharif", f)
    assert s.score == pytest.approx(min(s.factors.values()))
    mean = sum(s.factors.values()) / len(s.factors)
    assert s.score <= mean


def test_limiting_factor_is_the_argmin_and_is_explained():
    s = score("Rice", "Kharif", taluka("SOLAPUR", "SANGOLE"))
    assert s.factors[s.limiting_factor] == pytest.approx(s.score)
    assert s.limiting_factor in s.reason or "Vetoed" in s.reason


def test_suitability_class_thresholds():
    assert classify_score(0.80)[0] == "S1"
    assert classify_score(0.75)[0] == "S1"
    assert classify_score(0.60)[0] == "S2"
    assert classify_score(0.30)[0] == "S3"
    assert classify_score(0.10)[0] == "N"


# ------------------------------------------------- the vetoes that matter ----
def test_rice_is_vetoed_on_an_excessively_drained_shallow_upland():
    """The exact failure mode plan §3 names as the reason the gate exists."""
    f = taluka("AHILYANAGAR", "AKOLE")
    assert f["drainage_ord"] == 6 and f["depth_mm"] <= 300
    s = score("Rice", "Kharif", f)
    assert s.vetoed and s.suitability_class == "N"
    assert s.limiting_factor == "drainage"


def test_rice_is_marginal_not_impossible_in_a_semi_arid_taluka():
    """Solapur really does grow a little rice, so a hard veto would be wrong.

    Under the single 2025-26 weather year the gate vetoed rice at Sangole. The
    three-year normal (539 mm Kharif, range 369-725) puts it at S3 instead —
    and the APY panel vindicates that: Solapur plants 200-1,924 ha of Kharif
    rice every year of the panel, at 0.09-0.38 t/ha. Marginal is the correct
    verdict; impossible was not.
    """
    s = score("Rice", "Kharif", taluka("SOLAPUR", "SANGOLE"))
    assert not s.vetoed
    assert s.suitability_class == "S3"
    assert s.limiting_factor == "rain"
    # and it must still rank near the bottom, not be recommended
    ranked = score_all(taluka("SOLAPUR", "SANGOLE"), "Kharif")
    assert [x.crop for x in ranked].index("Rice") >= len(ranked) - 5


def test_sugarcane_is_vetoed_on_a_shallow_profile():
    """A 12-month crop needs 900 mm of profile; 300 mm cannot carry it."""
    s = score("Sugarcane", "Whole Year", taluka("AHILYANAGAR", "AKOLE"))
    assert s.vetoed
    # depth alone is below the N threshold, whichever factor ends up the argmin
    assert s.factors["depth"] < 0.25


def test_out_of_season_crops_are_vetoed_on_the_calendar():
    s = score("Wheat", "Kharif", taluka("NASHIK", "NIPHAD"))
    assert s.vetoed and s.limiting_factor == "season"
    assert "not grown in Kharif" in s.reason


def test_whole_year_admits_only_perennial_and_annual_duration_crops():
    got = {s.crop for s in score_all(taluka("KOLHAPUR", "KARVIR"), "Whole Year")}
    assert got == {"Sugarcane", "Turmeric", "Banana", "Grapes", "Pomegranate"}


# ------------------------------------------ regionally correct behaviour ----
def test_kolhapur_ranks_sugarcane_first():
    """Kolhapur is Maharashtra's sugarcane belt — 33-44% of district area, every
    year of the panel."""
    ranked = score_all(taluka("KOLHAPUR", "KARVIR"), "Whole Year")
    assert ranked[0].crop == "Sugarcane"
    assert ranked[0].suitability_class in {"S1", "S2"}
    assert ranked[0].score > 0.7


def test_semi_arid_solapur_rabi_favours_the_residual_moisture_crops():
    """Chickpea, safflower and Rabi wheat are what Solapur actually grows."""
    top = {s.crop for s in score_all(taluka("SOLAPUR", "SANGOLE"), "Rabi")[:5]}
    assert {"Chickpea", "Safflower"} <= top


def test_high_rainfall_hill_taluka_favours_millets_over_rice():
    ranked = score_all(taluka("AHILYANAGAR", "AKOLE"), "Kharif")
    assert ranked[0].crop in {"Finger Millet", "Maize", "Sorghum"}
    assert "Rice" not in {s.crop for s in ranked[:5]}


# ----------------------------------------------- the residual-moisture fix ---
def test_rabi_water_uses_the_stored_profile_not_just_in_season_rain():
    """Rabi rainfall is ~60 mm statewide; scoring against it alone vetoes
    every Rabi crop in Maharashtra, which is plainly wrong."""
    f = taluka("SOLAPUR", "SANGOLE")
    assert f["rain_Rabi"] < 150
    assert effective_water(f, "Rabi") == pytest.approx(f["rain_Rabi"] + f["rootzone_awc"])
    assert effective_water(f, "Kharif") == pytest.approx(f["rain_Kharif"])


def test_irrigation_leaves_almost_every_taluka_with_a_viable_crop():
    """A season the gate wipes out state-wide even under irrigation is a bug.

    Uran (Raigad) is the single genuine exception: coastal, poorly drained and
    saline, so no field crop in the envelope set clears it. That is correct.
    """
    for season in ("Kharif", "Rabi", "Summer"):
        viable = sum(
            any(not s.vetoed for s in score_all(row.to_dict(), season, irrigated=True))
            for _, row in FEATURES.iterrows()
        )
        assert viable >= 340, f"{season}: only {viable}/351 talukas viable under irrigation"


def test_rabi_viability_is_reported_as_a_water_need_not_a_veto():
    """Rabi runs on stored moisture and irrigation, not on Rabi rainfall.

    Deciding the veto on the rainfed score alone was wrong 38.7% of the time in
    Rabi, measured against eight years of practice. The veto now fires only on
    physical impossibility and a water shortfall is reported as
    ``requires_irrigation`` instead — a statement a farmer can act on.
    """
    viable = needs_water = 0
    for _, r in FEATURES.iterrows():
        crops = [s for s in score_all(r.to_dict(), "Rabi") if not s.vetoed]
        viable += bool(crops)
        needs_water += any(s.requires_irrigation for s in crops)
    assert viable >= 340          # almost everywhere can grow *something*
    assert needs_water > 0        # and much of it needs water supplied


def test_water_limited_talukas_are_flagged_as_needing_irrigation():
    """A taluka where most viable crops need water is told so, not left silent."""
    flagged = [r["Taluka"] for _, r in FEATURES.iterrows()
               if water_limited(r.to_dict(), "Rabi")]
    assert len(flagged) > 20
    for _, row in FEATURES.iterrows():
        f = row.to_dict()
        if water_limited(f, "Rabi"):
            viable = [s for s in score_all(f, "Rabi") if not s.vetoed]
            assert viable
            assert sum(s.requires_irrigation for s in viable) / len(viable) >= 0.5
            break


def test_veto_fires_only_on_physical_impossibility():
    """A water shortfall must never remove a crop; a shallow profile must."""
    for _, row in FEATURES.sample(40, random_state=config.SEED).iterrows():
        for season in ("Kharif", "Rabi"):
            for s in score_all(row.to_dict(), season):
                if s.vetoed:
                    # every veto names a hard factor, never rain or LGP alone
                    assert s.hard_limiting_factor is not None or \
                        s.score_irrigated < 0.25
                if s.requires_irrigation:
                    assert not s.vetoed


def test_long_duration_crops_are_not_vetoed_on_rainfed_growing_period():
    """Observed LGP spans 70-149 days, so a 150-day requirement vetoed
    sugarcane in every taluka in Maharashtra — across 96% of the area it
    actually occupies. LGP now gates only crops short enough to finish in a
    rainfed season."""
    kolhapur = taluka("KOLHAPUR", "KARVIR")
    cane = score("Sugarcane", "Whole Year", kolhapur)
    assert not cane.vetoed
    assert cane.factors["LGP"] == 1.0
    # a short-duration crop is still gated on it
    assert cr.get("Mungbean").duration_days < 150


def test_irrigation_only_relaxes_water_never_the_other_factors():
    f = taluka("AHILYANAGAR", "AKOLE")
    dry = score("Rice", "Kharif", f)
    wet = score("Rice", "Kharif", f, irrigated=True)
    assert dry.vetoed and wet.vetoed              # drainage is still fatal
    assert wet.factors["drainage"] == dry.factors["drainage"]


# --------------------------------------------------------------- the gate ---
def test_gate_separates_survivors_from_logged_vetoes():
    """Sugarcane is a Whole Year crop, so a Kharif query must veto it."""
    candidates = ["Rice", "Sorghum", "Pearl Millet", "Sugarcane"]
    survivors, vetoes = gate(candidates, "Kharif", taluka("SOLAPUR", "SANGOLE"))
    assert "Sugarcane" not in survivors
    assert "Sorghum" in survivors
    assert vetoes
    assert all(v.limiting_factor and v.reason for v in vetoes)


def test_gate_never_rejects_a_crop_it_cannot_assess():
    """Crops with no envelope pass through — abstention, not rejection."""
    survivors, vetoes = gate(["Other Cereals", "Niger seed"], "Kharif",
                             taluka("SOLAPUR", "SANGOLE"))
    assert survivors == ["Other Cereals", "Niger seed"]
    assert vetoes == []


def test_every_mapped_apy_crop_has_an_envelope():
    """Without this the gate silently abstains on crops it should be judging."""
    missing = [a for a in CROP_ONTOLOGY if cr.get(to_fertiliser_crop(a)) is None]
    assert missing == []


def test_envelopes_are_internally_ordered():
    for name, req in cr.CROP_REQUIREMENTS.items():
        for field in ("rain_mm", "temp_c", "ph", "drainage"):
            e = getattr(req, field)
            assert e.abs_min <= e.opt_min <= e.opt_max <= e.abs_max, (name, field)
