"""The farmer-facing invariants: each input must move the answer the right way.

The relations themselves live in ``src/eval/metamorphic.py`` so the scorecard
can run the full sweep — one taluka per district, both seasons. Here they run
on a small sample, because a relation that only holds on the sample the report
happens to use is not an invariant.

Each is a statement a farmer would recognise:

    M1  a card reading acid soil never *promotes* a crop that wants lime
    M2  a card reading saline never promotes a salt-sensitive crop
    M3  saying "I can irrigate" never rules out a crop that rainfed allows
    M4  asking for Rabi, irrigated, gets the Rabi irrigated recipe
    M6  a card low in nitrogen gets more nitrogen, not less
"""
import pytest

from src.eval import metamorphic as mm


@pytest.fixture(scope="module")
def pipe():
    from src.pipeline import load_pipeline
    return load_pipeline()


@pytest.fixture(scope="module")
def sweep(pipe):
    """Two districts, both seasons — every relation, on the served path."""
    checks, recs = mm.run(pipe, mm.sample_talukas(2), seasons=("Kharif", "Rabi"))
    return checks, recs


def _assert_relation(checks, name: str) -> None:
    sub = checks[(checks["test"] == name) & checks["passed"].notna()]
    assert len(sub), f"{name} produced no applicable checks — the relation is not being tested"
    bad = sub[~sub["passed"].astype(bool)]
    assert bad.empty, "\n".join(
        f"{r.taluka}/{r.season} {r.crop or ''}: {r.detail}" for r in bad.itertuples())


def test_m1_acid_soil_never_promotes_a_crop_that_wants_lime(sweep):
    _assert_relation(sweep[0], "M1")


def test_m2_saline_soil_never_promotes_a_salt_sensitive_crop(sweep):
    _assert_relation(sweep[0], "M2")


def test_m3_irrigation_never_rules_out_what_rainfed_allows(sweep):
    """The one direction a water scenario must never move the answer."""
    _assert_relation(sweep[0], "M3")


def test_m4_the_recipe_matches_the_season_and_water_regime_asked_for(sweep):
    _assert_relation(sweep[0], "M4")


def test_m6_a_poorer_card_gets_a_larger_dose(sweep):
    _assert_relation(sweep[0], "M6")


def test_m5_a_photograph_never_lifts_a_hard_factor_veto(sweep):
    """M5 was a placeholder while the engine took no image. It now has teeth.

    This test used to assert the opposite — that M5 reported itself untested —
    which was the right guard at the time: a relation that cannot fail is worse
    than an absent one, because it reads as evidence. Now that a photograph is
    an input, the guard is inverted. M5 runs every class the classifier knows at
    maximum confidence against every sampled taluka, and the set of crops vetoed
    on a *hard* factor (depth, drainage, pH, salinity, season, temp) must come
    out identical with and without the picture.

    That property is what makes it safe to let a photograph into the answer at
    all: a misread one can cost a farmer a recommendation, never their safety.
    """
    m5 = sweep[0][sweep[0]["test"] == "M5"]
    assert len(m5) > 1, "M5 ran no checks — it has slipped back to being a placeholder"
    assert m5["passed"].notna().all(), "M5 left a check untested"
    _assert_relation(sweep[0], "M5")


def test_every_farmer_input_is_measured_and_the_card_ones_all_bite(pipe):
    """An input the engine ignores is a promise the product does not keep.

    The photograph is now measured here like the rest — `input_effects` runs it
    rather than declaring it ineffective by inspecting `recommend`'s signature,
    which is what it did while the engine took no image.

    It is still permitted to come out ineffective on a two-district sample, and
    that is not a bug: fusion abstains where the survey maps a single soil type
    for the taluka, and a sample this small can easily contain only those. What
    is no longer permitted is for it to be *assumed* ineffective without being
    run.
    """
    effects = mm.input_effects(pipe, mm.sample_talukas(2))
    ineffective = effects[~effects["effective"].fillna(False).astype(bool)]
    assert set(ineffective["input"]) <= {"soil photo"}, \
        f"inputs that changed nothing: {sorted(ineffective['input'])}"
    assert len(effects) == 15
