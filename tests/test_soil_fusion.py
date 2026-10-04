"""What a photograph may and may not do to a recommendation.

The rules under test are not arbitrary. A farmer's photograph is one frame of
one field; the soil survey is a systematic map whose primary type covers 81% of
a taluka on average. The photograph is allowed to inform the answer and is not
allowed to overrule the map, and every test here pins one edge of that.
"""
from __future__ import annotations

import json

import pytest

from src.rules import soil_fusion as sf


@pytest.fixture
def metadata(tmp_path):
    """A classifier whose held-out behaviour is good but not perfect.

    Rows are truth, columns prediction, in the order Alluvial / Black / Clay /
    Red — the owner's four classes. The off-diagonal mass is what stops one
    photograph from settling the question on its own.
    """
    path = tmp_path / "soil_metadata.json"
    path.write_text(json.dumps({
        "classes": ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"],
        "confusion_pooled": [
            [240, 5, 20, 2],      # truth Alluvial
            [4, 105, 8, 3],       # truth Black
            [6, 2, 100, 3],       # truth Clay
            [3, 2, 5, 136],       # truth Red
        ],
    }))
    sf._likelihood.cache_clear()
    return str(path)


def survey(primary, secondary=None, share=70.0):
    return {"soil_type": primary, "soil_type_secondary": secondary,
            "share_pct": share, "texture": "clayey"}


def photo(label, confidence):
    """A calibrated prediction: the named class takes `confidence`, rest split."""
    classes = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]
    rest = (1.0 - confidence) / (len(classes) - 1)
    return {c: (confidence if c == label else rest) for c in classes}


class TestWhenTheSurveyStands:
    def test_an_unconfident_photograph_changes_nothing(self, metadata):
        out = sf.fuse(survey("Black (Regur)", "Alluvial"),
                      photo("Alluvial soil", 0.55), metadata)
        assert not out.applied
        assert out.soil_type == "Black (Regur)"
        assert "below" in out.reason

    def test_a_photograph_of_something_that_is_not_soil_changes_nothing(self, metadata):
        out = sf.fuse(survey("Black (Regur)", "Alluvial"),
                      photo("Alluvial soil", 0.99), metadata, in_distribution=False)
        assert not out.applied
        assert out.soil_type == "Black (Regur)"

    def test_a_single_soil_taluka_has_nothing_to_choose_between(self, metadata):
        """A confident photograph cannot introduce a soil the survey never placed here."""
        out = sf.fuse(survey("Black (Regur)"), photo("Alluvial soil", 0.99), metadata)
        assert not out.applied
        assert out.soil_type == "Black (Regur)"

    def test_it_abstains_where_the_classifier_has_no_class(self, metadata):
        """Laterite is a real Konkan soil and the owner's data has 29 images of it.

        The classifier therefore cannot recognise it, and the honest response in
        a laterite taluka is to abstain rather than to pick whichever of its four
        classes happens to score highest.
        """
        out = sf.fuse(survey("Laterite", "Red & Yellow"),
                      photo("Red soil", 0.99), metadata)
        assert not out.applied
        assert "no class for Laterite" in out.reason

    def test_no_photograph_leaves_the_survey_alone(self, metadata):
        out = sf.fuse(survey("Black (Regur)", "Alluvial"), None, metadata)
        assert not out.applied
        assert out.soil_type == "Black (Regur)"


class TestWhenThePhotographActs:
    def test_a_confident_photograph_can_choose_the_secondary_soil(self, metadata):
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=55.0),
                      photo("Alluvial soil", 0.97), metadata)
        assert out.applied
        assert out.soil_type == "Alluvial"
        assert out.posterior["Alluvial"] > out.posterior["Black (Regur)"]

    def test_a_dominant_survey_resists_a_single_photograph(self, metadata):
        """95/5 is the survey saying the taluka is nearly uniform.

        One photograph of the 5% should not flip it; that is the prior doing
        exactly the job it is there for.
        """
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=95.0),
                      photo("Alluvial soil", 0.90), metadata)
        assert out.soil_type == "Black (Regur)"
        assert not out.applied

    def test_the_posterior_is_a_distribution(self, metadata):
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=60.0),
                      photo("Alluvial soil", 0.95), metadata)
        assert out.posterior
        assert sum(out.posterior.values()) == pytest.approx(1.0)


class TestClayIsNotASoilType:
    def test_clay_casts_no_vote(self, metadata):
        """Clay is a texture, not a soil order, and no survey type corresponds.

        A confident Clay photograph must not pick between Black and Alluvial,
        because it is not evidence about which of them is underfoot.
        """
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=55.0),
                      photo("Clay soil", 0.99), metadata)
        assert out.soil_type == "Black (Regur)"
        assert not out.applied

    def test_clay_is_absent_from_the_likelihood_table(self, metadata):
        table, _classes = sf._likelihood(metadata)
        assert "Clay" not in table
        assert set(table) == {"Alluvial", "Black (Regur)", "Red & Yellow"}


class TestEitherClassifierVocabulary:
    """Two classifiers can be deployed, and they spell their classes differently.

    The old checkpoint says `alluvial`/`black`/`red`; one trained on the owner's
    folders says `Alluvial soil`/`Black Soil`/`Red soil`. Matching literally
    would make fusion abstain silently against whichever it was not written
    for — the photograph would stop mattering and nothing would report it.
    """

    @pytest.fixture
    def old_metadata(self, tmp_path):
        """The eight-class checkpoint: lowercase names, best-fold matrix only."""
        path = tmp_path / "old_metadata.json"
        classes = ["alluvial", "black", "cinder", "clay",
                   "laterite", "peat", "red", "yellow"]
        matrix = [[0] * 8 for _ in range(8)]
        for i in range(8):
            matrix[i][i] = 40
            matrix[i][(i + 1) % 8] = 4
        path.write_text(json.dumps(
            {"classes": classes, "confusion_best_fold": matrix}))
        sf._likelihood.cache_clear()
        return str(path)

    def test_it_reads_the_older_best_fold_matrix(self, old_metadata):
        """The old metadata has no `confusion_pooled`; refusing it would mean
        fusion could never run against the model that is actually shipped."""
        table, _classes = sf._likelihood(old_metadata)
        assert table, "fusion found no likelihood in the old metadata"

    def test_lowercase_class_names_still_map_to_survey_types(self, old_metadata):
        table, _classes = sf._likelihood(old_metadata)
        assert "Black (Regur)" in table
        assert "Alluvial" in table

    def test_red_and_yellow_are_one_surveyed_soil(self, old_metadata):
        """The old list separates them; the Maharashtra survey does not."""
        table, _classes = sf._likelihood(old_metadata)
        assert "Red & Yellow" in table
        # both classifier columns feed the one surveyed type
        assert table["Red & Yellow"]["red"] > 0
        assert table["Red & Yellow"]["yellow"] > 0

    def test_non_maharashtra_classes_cast_no_vote(self, old_metadata):
        """Cinder and peat are artefacts of the old web-scraped class list."""
        table, _classes = sf._likelihood(old_metadata)
        assert "cinder" not in table
        assert "peat" not in table

    def test_a_photograph_from_the_old_model_can_still_act(self, old_metadata):
        probabilities = {c: 0.01 for c in
                         ["alluvial", "black", "cinder", "clay",
                          "laterite", "peat", "red", "yellow"]}
        probabilities["alluvial"] = 0.93
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=55.0),
                      probabilities, old_metadata)
        assert out.soil_type == "Alluvial"
        assert out.applied

    def test_normalise_collapses_the_spellings(self):
        assert sf.normalise("Black Soil") == sf.normalise("black") == "black"
        assert sf.normalise("Alluvial soil") == "alluvial"


class TestHardFactorsAreUntouchable:
    """M5: a photograph may add a constraint and may never remove one."""

    def test_depth_drainage_and_salinity_never_move(self, metadata):
        feats = {"depth_mm": 450.0, "drainage_ord": 2.0, "ec_saline": 12.0,
                 "rootzone_awc": 120.0, "is_black_soil": 1, "is_lateritic": 0}
        out = sf.fuse(survey("Black (Regur)", "Alluvial", share=55.0),
                      photo("Alluvial soil", 0.97), metadata)
        applied = sf.apply_to_features(feats, out)

        assert out.applied, "this fixture is meant to exercise an applied fusion"
        for hard in ("depth_mm", "drainage_ord", "ec_saline"):
            assert applied[hard] == feats[hard]

    def test_it_updates_the_soil_type_flags_the_gate_reads(self, metadata):
        feats = {"rootzone_awc": 100.0, "is_black_soil": 1, "is_lateritic": 0}
        out = sf.fuse(survey("Black (Regur)", "Red & Yellow", share=55.0),
                      photo("Red soil", 0.97), metadata)
        applied = sf.apply_to_features(feats, out)

        assert out.soil_type == "Red & Yellow"
        assert applied["is_black_soil"] == 0
        assert applied["is_lateritic"] == 1

    def test_an_unapplied_fusion_returns_the_features_untouched(self, metadata):
        feats = {"rootzone_awc": 100.0, "is_black_soil": 1, "is_lateritic": 0}
        out = sf.fuse(survey("Black (Regur)"), photo("Red soil", 0.99), metadata)
        assert sf.apply_to_features(feats, out) == feats
