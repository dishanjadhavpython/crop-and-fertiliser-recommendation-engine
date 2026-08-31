"""The S2 gate, validated against eight years of revealed practice.

The gate uses no labels, so it was never checked against anything. These tests
make it falsifiable and guard the two failures the validation actually found:
a length-of-growing-period requirement no taluka in Maharashtra could satisfy,
and a veto that fired on water shortfall rather than on impossibility.
"""
import pandas as pd
import pytest

from src import config
from src.eval.gate_validation import (
    agreement_table, factor_blame, monotonicity, score_panel, veto_metrics,
)
from src.rules import crop_requirements as cr
from src.rules.suitability import LGP_MAX_DURATION_DAYS, score


@pytest.fixture(scope="module")
def scored():
    return score_panel()


def test_gate_agrees_with_what_farmers_plant(scored):
    """A crop the gate calls suitable must be planted far more than one it vetoes."""
    m = monotonicity(scored)
    assert m["suitable_vs_vetoed_planted_ratio"] >= 4.0


def test_grade_ordering_is_not_inverted(scored):
    """Mean area share must fall S1 -> S2 -> S3 -> N.

    Before the hard/soft split it rose (0.070 / 0.103 / 0.146), meaning the
    four-level grade ran backwards against practice.
    """
    m = monotonicity(scored)
    assert m["area_share_monotone_decreasing"]
    assert m["planted_rate_monotone_decreasing"]


def test_false_veto_rate_stays_low(scored):
    """The regression guard. A veto removes a crop outright, so a false veto is
    the most damaging error the system can make. Was 8.24% / 6.34%."""
    m = veto_metrics(scored)
    assert m["false_veto_rate_area_gt_0.01"] <= 0.02
    assert m["false_veto_rate_area_gt_0.05"] <= 0.015


def test_agronomic_veto_rate_is_small(scored):
    """Excluding the season veto — which is legitimate and 0.4% wrong — the
    gate should reject only a small slice of the candidate space."""
    agronomic = scored["vetoed"] & (scored["limiting_factor"] != "season")
    assert float(agronomic.mean()) <= 0.05


def test_season_veto_is_the_reliable_one(scored):
    """It is 79% of all vetoes and should be almost never wrong."""
    blame = factor_blame(scored).set_index("limiting_factor")
    assert blame.loc["season", "false_veto_rate"] < 0.02


def test_lgp_no_longer_vetoes_anything_it_should_not(scored):
    """LGP produced a 79% false-veto rate before the duration cap."""
    blame = factor_blame(scored).set_index("limiting_factor")
    if "LGP" in blame.index:
        assert blame.loc["LGP", "false_veto_rate"] < 0.2


def test_sugarcane_is_never_lgp_vetoed():
    """Observed LGP spans 70-149 days; sugarcane required 150, so it was
    vetoed in every taluka in the state across 96% of the area it occupies."""
    store = pd.read_parquet(config.FEATURES / "taluka_features.parquet")
    assert store["lgp_annual"].max() < cr.get("Sugarcane").min_lgp_days
    for _, row in store.sample(25, random_state=config.SEED).iterrows():
        s = score("Sugarcane", "Whole Year", row.to_dict())
        assert s.factors.get("LGP") == 1.0
        assert s.limiting_factor != "LGP"


def test_lgp_still_gates_short_duration_crops():
    """The cap must not switch the factor off wholesale."""
    assert cr.get("Mungbean").duration_days <= LGP_MAX_DURATION_DAYS
    assert cr.get("Sugarcane").duration_days > LGP_MAX_DURATION_DAYS
    assert cr.get("Tetraploid Cotton").duration_days > LGP_MAX_DURATION_DAYS


def test_water_shortfall_is_reported_not_vetoed(scored):
    """A crop failing only on water needs irrigation; it is not impossible."""
    needs = scored[scored["requires_irrigation"]]
    assert len(needs) > 500
    assert not needs["vetoed"].any()
    # and the flag is meaningful: these are planted far more than average
    assert needs["planted"].mean() > scored["planted"].mean()


def test_envelopes_carry_provenance():
    """Every encoded crop names a species and an authority."""
    for name in cr.CROP_REQUIREMENTS:
        meta = cr.CROP_SOURCES.get(name)
        assert meta and meta["species"] and meta["source"], name


def test_verification_status_is_stated_honestly():
    """VERIFIED_AGAINST_SOURCE must only ever contain genuinely checked crops."""
    assert cr.VERIFIED_AGAINST_SOURCE <= set(cr.CROP_REQUIREMENTS)
