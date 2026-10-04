"""End-to-end pipeline behaviour (plan §3).

The stage ordering is load-bearing and these tests pin it down:
S1 proposes and S2 vetoes; S4 stays exact regardless of S1's confidence; and
S5 falls back to rules rather than to nonsense when it is out of its depth.

The pipeline is fitted once for the module — training takes ~30 s.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src import config
from src.pipeline import recommend, train
from src.rules.soil_class import SoilTest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def pipe():
    return train(seeds=2, verbose=False)


# ------------------------------------------------------------- structure ----
def test_returns_a_ranked_list_not_a_single_crop(pipe):
    """A farmer needs a shortlist with reasons, not an argmax (plan §5.1)."""
    rec = recommend("KOLHAPUR", "KARVIR", "Kharif", pipe=pipe, top_k=5)
    assert len(rec.crops) == 5
    assert [c.rank for c in rec.crops] == [1, 2, 3, 4, 5]
    scores = [c.final_score for c in rec.crops]
    assert scores == sorted(scores, reverse=True)


def test_every_recommendation_carries_a_reason(pipe):
    rec = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe)
    for c in rec.crops:
        assert c.reason and len(c.reason) > 20
        assert c.limiting_factor
        assert c.decided_by in {"model", "rules", "blend", "rules (out of distribution)"}


def test_unknown_taluka_is_rejected_not_guessed(pipe):
    with pytest.raises(KeyError):
        recommend("KOLHAPUR", "NOWHERE", "Kharif", pipe=pipe)


# --------------------------------------------- S1 proposes, S2 vetoes -------
def test_the_gate_removes_agronomically_impossible_crops(pipe):
    """No vetoed crop may survive into the recommendation list."""
    for district, taluka, season in (("AHILYANAGAR", "AKOLE", "Kharif"),
                                     ("SOLAPUR", "SANGOLE", "Kharif"),
                                     ("KOLHAPUR", "KARVIR", "Whole Year")):
        rec = recommend(district, taluka, season, pipe=pipe, top_k=8)
        recommended = {c.crop for c in rec.crops}
        vetoed = {v["crop"] for v in rec.vetoed}
        assert not (recommended & vetoed)


def test_rice_is_never_recommended_on_an_excessively_drained_upland(pipe):
    """The exact failure mode the gate exists to prevent (plan §3)."""
    rec = recommend("AHILYANAGAR", "AKOLE", "Kharif", pipe=pipe, top_k=10)
    assert "Rice" not in {c.crop for c in rec.crops}
    assert "Rice" in {v["crop"] for v in rec.vetoed}


def test_every_veto_is_logged_with_its_limiting_factor(pipe):
    """That log is the explanation layer and the debugging tool."""
    rec = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe)
    assert rec.vetoed
    for v in rec.vetoed:
        assert v["limiting_factor"] and v["reason"]


def test_agronomic_safety_rate_is_zero_across_the_state(pipe):
    """Plan §7.5: recommendations violating a hard constraint. Target: zero."""
    store = pd.read_parquet(config.FEATURES / "taluka_features.parquet")
    violations = 0
    for _, row in store.sample(25, random_state=config.SEED).iterrows():
        for season in ("Kharif", "Rabi"):
            rec = recommend(row["District"], row["Taluka"], season, pipe=pipe, top_k=5)
            vetoed = {v["crop"] for v in rec.vetoed}
            violations += len({c.crop for c in rec.crops} & vetoed)
    assert violations == 0


# ------------------------------------------------------- S4 independence ----
def test_fertiliser_advice_is_exact_even_where_crop_advice_is_uncertain(pipe):
    """S4 depends on S1's output but not on its confidence (plan §3)."""
    from src.rules import fertiliser as fz

    rec = recommend("KOLHAPUR", "KARVIR", "Whole Year", pipe=pipe, top_k=3)
    for c in rec.crops:
        f = c.fertiliser
        if not (f and f.get("available") and not f.get("estimated")):
            continue
        direct = fz.lookup(rec.district, _fert_name(c.crop), f["soil_class"])
        assert f["table_products_kg_ha"] == direct["products"]


def test_farmer_soil_test_overrides_the_taluka_distribution(pipe):
    default = recommend("NASHIK", "NIPHAD", "Rabi", pipe=pipe)
    supplied = recommend("NASHIK", "NIPHAD", "Rabi", pipe=pipe,
                         soil_test=SoilTest(700, 40, 350, 1.0))
    assert default.soil_test_source == "taluka SHC distribution"
    # This card carried N/P/K/OC and nothing else, and the answer says so
    # rather than claiming the whole card: the other eight readings are the
    # taluka's, and each one now carries its own provenance.
    assert supplied.soil_test_source.startswith("farmer soil health card")
    assert supplied.context["soil_test_sources"]["N"] == "farmer soil health card"
    assert supplied.context["soil_test_sources"]["Zn"] == "taluka SHC distribution"
    assert supplied.soil_class == "High"


def test_a_richer_soil_receives_a_smaller_nitrogen_dose(pipe):
    """The whole point of soil-test-based recommendation."""
    poor = recommend("NASHIK", "NIPHAD", "Rabi", pipe=pipe,
                     soil_test=SoilTest(150, 5, 70, 0.2), top_k=6)
    rich = recommend("NASHIK", "NIPHAD", "Rabi", pipe=pipe,
                     soil_test=SoilTest(700, 40, 350, 1.0), top_k=6)
    poor_n = {c.crop: c.fertiliser["interpolated_target_kg_ha"]["N"]
              for c in poor.crops if c.fertiliser and c.fertiliser.get("available")}
    rich_n = {c.crop: c.fertiliser["interpolated_target_kg_ha"]["N"]
              for c in rich.crops if c.fertiliser and c.fertiliser.get("available")}
    shared = set(poor_n) & set(rich_n)
    assert shared
    assert all(rich_n[c] < poor_n[c] for c in shared)


def test_micronutrient_layer_reaches_the_output(pipe):
    """The six components the government table ignores (plan §6.3)."""
    rec = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe)
    assert rec.micronutrients
    assert {m["component"] for m in rec.micronutrients} <= set(config.MICRONUTRIENTS)


# ------------------------------------------- the other eight SHC components -
# N/P/K/OC are the four the fertility table indexes. The card carries twelve;
# these tests cover the remaining eight (pH, EC, and six micronutrients)
# reaching the pipeline the same way -- a farmer's own reading overriding the
# taluka average, never silently ignored.
def test_farmer_ph_and_ec_override_the_taluka_average(pipe):
    default = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe)
    supplied = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe,
                         soil_test=SoilTest(ph=6.8, ec_status="normal"))
    # Without a card the pH is the soil survey's own class midpoint, not a Soil
    # Health Card statistic — the label used to say SHC, which was the one
    # provenance claim in the response that was not true.
    assert default.context["ph_source"] == "taluka soil survey pH class"
    assert supplied.context["ph_source"] == "farmer soil health card"
    assert supplied.context["ph_used"] == pytest.approx(6.8)
    assert supplied.context["ec_source"] == "farmer soil health card"
    assert supplied.context["ec_saline_pct_used"] == pytest.approx(0.0)


def test_extreme_farmer_ph_and_salinity_can_veto_a_crop_the_taluka_average_would_not(pipe):
    """Proves the override reaches the Liebig gate, not just the response payload.

    The vetoes here come from the pH. A card's EC "high" no longer vetoes on its
    own — it cannot place the field in the survey's saline class — so it is
    asserted separately below, as a caution rather than a gate.
    """
    mild = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe, top_k=10)
    hostile = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe, top_k=10,
                        soil_test=SoilTest(ph=9.4, ec_status="high"))
    mild_crops = {c.crop for c in mild.crops}
    hostile_crops = {c.crop for c in hostile.crops}
    assert hostile_crops < mild_crops
    hostile_vetoed = {v["crop"] for v in hostile.vetoed}
    assert hostile_vetoed - {v["crop"] for v in mild.vetoed}


def test_the_card_reaches_the_learned_score_not_only_the_gate(pipe):
    """Two farmers in one district with different soil must get different rankings.

    The agronomic-fit block is computed from the taluka and the farmer's own
    readings, so a card changes the learned score itself. Before that, the
    twelve readings reached the Liebig gate and the dose but never S1 — the
    ranker saw a district average and nothing else, and every farmer in the
    district got the same ordered list.

    The pH contrast here is deliberately mild. At pH 5 almost every crop is
    vetoed outright and the two answers share no crops at all — which proves
    the card matters, but says nothing about the *learned* score, which is what
    this test is for. The crop-set effect is asserted separately below.
    """
    mild = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe, top_k=12,
                     soil_test=SoilTest(ph=6.2))
    alkaline = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe, top_k=12,
                         soil_test=SoilTest(ph=7.6))

    a = {c.crop: c.learned_score for c in mild.crops if c.learned_score is not None}
    b = {c.crop: c.learned_score for c in alkaline.crops if c.learned_score is not None}
    shared = set(a) & set(b)
    assert shared, "no crop appears in both answers"
    assert any(abs(a[c] - b[c]) > 1e-6 for c in shared), \
        "the farmer's pH did not move a single learned score"


def test_an_extreme_card_changes_which_crops_are_offered(pipe):
    """The same mechanism at the other end of the scale.

    pH 5 rules out what pH 8.6 allows, so the two farmers are not shown the
    same shortlist — the outcome a farmer would actually notice.
    """
    acid = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe, top_k=12,
                     soil_test=SoilTest(ph=5.0))
    alkaline = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe, top_k=12,
                         soil_test=SoilTest(ph=8.6))
    assert {c.crop for c in acid.crops} != {c.crop for c in alkaline.crops}


def test_a_soil_test_with_only_ph_does_not_crash_on_missing_npk(pipe):
    """A partial OCR read -- pH confirmed, N/P/K/OC not -- is a real case."""
    rec = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe,
                    soil_test=SoilTest(ph=6.9))
    assert rec.context["ph_source"] == "farmer soil health card"
    assert rec.context["ph_used"] == pytest.approx(6.9)
    assert rec.soil_class in config.SOIL_CLASSES   # backfilled from the taluka, not crashed


def test_farmer_micronutrient_verdicts_reach_the_pipeline_output(pipe):
    rec = recommend("SOLAPUR", "SANGOLE", "Rabi", pipe=pipe,
                    soil_test=SoilTest(n_kg_ha=400, p_kg_ha=17, k_kg_ha=190, oc_pct=0.6,
                                       micronutrients={"Zn": "low", "Fe": "normal"}))
    by_component = {m["component"]: m for m in rec.micronutrients}
    assert by_component["Zn"]["source"] == "farmer soil health card"
    assert "Fe" not in by_component   # farmer says sufficient -- suppressed regardless of taluka share


# ------------------------------------------------------- S3 and S5 ----------
def test_yield_is_reported_as_a_band_never_a_point(pipe):
    """R2 = 0.19 makes a point estimate dishonest (plan §5.3)."""
    rec = recommend("KOLHAPUR", "KARVIR", "Kharif", pipe=pipe)
    for c in rec.crops:
        if c.yield_p50_t_ha is None:
            continue
        assert c.yield_p10_t_ha <= c.yield_p50_t_ha <= c.yield_p90_t_ha
        assert c.yield_p90_t_ha > c.yield_p10_t_ha
        assert c.yield_p10_t_ha >= 0
        assert "band" in c.yield_interval_note


def test_yield_bands_are_on_the_crops_own_scale(pipe):
    """Converted back per crop from the within-crop z-score (plan §5.3)."""
    rec = recommend("KOLHAPUR", "KARVIR", "Whole Year", pipe=pipe, top_k=3)
    cane = next((c for c in rec.crops if c.crop == "Sugarcane"), None)
    assert cane is not None
    assert 20 < cane.yield_p50_t_ha < 200      # sugarcane is tens of t/ha
    rec2 = recommend("SOLAPUR", "SANGOLE", "Kharif", pipe=pipe, top_k=8)
    sesame = next((c for c in rec2.crops if c.crop == "Sesamum"), None)
    if sesame:
        assert sesame.yield_p50_t_ha < 3       # sesamum is well under 1 t/ha


def test_alpha_comes_from_ranking_skill_not_yield_skill(pipe):
    """The alpha weights govern S1, which is a ranker (§5.4).

    The first implementation derived them from the *yield* model's within-crop
    Spearman. That zeroed cotton, gram, tur and safflower and handed them to
    the rule scorer — costing 0.098 NDCG@5 — because the yield model fails on
    those crops even though the ranker ranks them well.
    """
    assert pipe.rank_skill is not None and pipe.yield_skill is not None
    rank = pipe.rank_skill.set_index("Crop")["rho"]
    yld = pipe.yield_skill.set_index("Crop")["rho"]

    # The two skills are different measurements of different models, and the
    # alphas must track the RANKING one. Asserted as a correlation over all
    # crops rather than on named crops, because which crops disagree is a
    # property of the data and moved once the panel grew to eight years.
    common = [c for c in pipe.alphas if c in rank.index and np.isfinite(rank[c])]
    assert len(common) > 10
    alpha = pd.Series({c: pipe.alphas[c] for c in common})
    assert alpha.corr(rank[common], method="spearman") > 0.8

    if len(set(common) & set(yld.index)) > 10:
        shared = [c for c in common if c in yld.index and np.isfinite(yld[c])]
        # the two skills are genuinely different signals, not interchangeable
        assert rank[shared].corr(yld[shared], method="spearman") < 0.95

    assert max(pipe.alphas.values()) >= 1.0


def test_blend_still_routes_around_the_crops_the_ranker_fails_on(pipe):
    rank = pipe.rank_skill.set_index("Crop")["rho"]
    for crop, alpha in pipe.alphas.items():
        if alpha == 0.0 and crop in rank.index and np.isfinite(rank[crop]):
            assert rank[crop] <= config.ALPHA_RHO_FLOOR


def test_irrigation_expands_what_is_recommendable(pipe):
    dry = recommend("SOLAPUR", "AKKALKOT", "Rabi", pipe=pipe, top_k=8)
    wet = recommend("SOLAPUR", "AKKALKOT", "Rabi", pipe=pipe, top_k=8, irrigated=True)
    assert len(wet.crops) >= len(dry.crops)
    assert len(wet.vetoed) <= len(dry.vetoed)


def test_water_limited_talukas_are_flagged(pipe):
    rec = recommend("SOLAPUR", "AKKALKOT", "Rabi", pipe=pipe)
    assert rec.water_limited is True
    assert recommend("KOLHAPUR", "KARVIR", "Kharif", pipe=pipe).water_limited is False


def test_output_serialises_to_json(pipe):
    import json
    rec = recommend("KOLHAPUR", "KARVIR", "Kharif", pipe=pipe)
    payload = json.loads(rec.to_json())
    assert payload["district"] == "KOLHAPUR"
    assert len(payload["crops"]) == len(rec.crops)


def test_marathi_names_are_served(pipe):
    rec = recommend("KOLHAPUR", "KARVIR", "Whole Year", pipe=pipe, top_k=3)
    named = [c for c in rec.crops if c.crop_marathi]
    assert named
    assert any(ord(ch) > 2300 for ch in named[0].crop_marathi)   # Devanagari


def _fert_name(apy_crop: str) -> str:
    from src.ontology.crop_map import to_fertiliser_crop
    return to_fertiliser_crop(apy_crop)


def test_unassessable_crops_do_not_compete_for_a_slot(pipe):
    """The seven APY aggregates and tobacco have no envelope and no recipe.

    Listing them as recommendations would pad the answer with rows the system
    has nothing to say about, so they are reported separately.
    """
    rec = recommend("KOLHAPUR", "KARVIR", "Kharif", pipe=pipe, top_k=5)
    listed = {c.crop for c in rec.crops}
    unassessable = {x["crop"] for x in rec.not_assessable}
    assert "Tobacco" in unassessable
    assert {"Other Cereals", "Other Kharif pulses"} <= unassessable
    assert not (listed & unassessable)
    for x in rec.not_assessable:
        assert x["reason"]


def test_no_yield_band_is_invented_for_a_crop_with_no_spread(pipe):
    """Tobacco is recorded once in all of Maharashtra — it has no estimable band."""
    rec = recommend("KOLHAPUR", "KARVIR", "Whole Year", pipe=pipe, top_k=5)
    for c in rec.crops:
        if c.yield_p50_t_ha is not None:
            assert c.yield_p90_t_ha > c.yield_p10_t_ha    # never a degenerate band
            assert c.yield_p50_t_ha > 0


def test_card_ec_high_cautions_instead_of_vetoing_every_crop(pipe):
    """A card EC a whisker over its printed range must not empty the list.

    It used to be read as "100% of samples saline", which zeroed the salinity
    factor for every crop — the fixture card (EC 1.06 against 0.2-0.9) got no
    crops at all in Baramati, salt-tolerant sorghum included.
    """
    plain = recommend("PUNE", "BARAMATI", "Rabi", pipe=pipe, soil_test=SoilTest(ph=8.12))
    high = recommend("PUNE", "BARAMATI", "Rabi", pipe=pipe,
                     soil_test=SoilTest(ph=8.12, ec_status="high"))
    assert high.crops, "an EC above the card's range must not veto every crop"
    assert {c.crop for c in high.crops} == {c.crop for c in plain.crops}
    assert high.context["ec_card_high"] is True
    assert set(high.context["ec_least_tolerant"]) <= {c.crop for c in high.crops}
    # Market names reach the agronomic table: Gram is Chickpea (18%), below the cut.
    if "Gram" in {c.crop for c in high.crops}:
        assert "Gram" in high.context["ec_least_tolerant"]
    normal = recommend("PUNE", "BARAMATI", "Rabi", pipe=pipe,
                       soil_test=SoilTest(ph=8.12, ec_status="normal"))
    assert normal.context["ec_card_high"] is False
    assert normal.context["ec_saline_pct_used"] == pytest.approx(0.0)
