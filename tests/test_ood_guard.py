"""The out-of-distribution guard (plan §5.4, review item 2).

The first implementation was fitted on the same 34 district vectors it then
scored, so nothing was ever out-of-distribution and it fired on 0 of 30
queries — decorative rather than protective. These tests pin down that it now
discriminates, and that it does not abstain on talukas it was trained on.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.features.build_store import build_feature_store, feature_columns
from src.models.conformal import MahalanobisGuard

KONKAN = {"RATNAGIRI", "SINDHUDURG", "RAIGAD", "PALGHAR", "THANE"}
STORE = build_feature_store()
COLS = [c for c in feature_columns(STORE) if STORE[c].notna().all()][:40]


def test_guard_detects_a_region_it_never_saw():
    """Fitted on inland Maharashtra, it must flag the Konkan far more often.

    This is the test that separates a working guard from a decorative one: the
    Konkan is genuinely a different agro-ecology (3,000+ mm, lateritic,
    poorly drained), and a guard that cannot see that cannot see anything.
    """
    train = STORE[~STORE["District"].isin(KONKAN)]
    held_out = STORE[STORE["District"].isin(KONKAN)]
    assert len(held_out) > 30

    guard = MahalanobisGuard(quantile=0.98).fit(train[COLS])
    inland_rate = guard.is_outlier(train[COLS]).mean()
    konkan_rate = guard.is_outlier(held_out[COLS]).mean()

    assert inland_rate < 0.05                      # calibrated as designed
    assert konkan_rate > 0.20                      # and the held-out region trips it
    assert konkan_rate > 5 * inland_rate           # a large enrichment, not noise


def test_guard_does_not_abstain_on_its_own_training_corpus():
    """Every served taluka IS training data, so none of them is novel.

    A guard that abstains on Kolhapur is not being careful, it is being wrong.
    """
    guard = MahalanobisGuard(quantile=1.0).fit(STORE[COLS])
    assert guard.is_outlier(STORE[COLS]).sum() == 0


def test_guard_flags_input_outside_the_corpus():
    guard = MahalanobisGuard(quantile=1.0).fit(STORE[COLS])
    absurd = STORE[COLS].iloc[[0]].copy()
    absurd.iloc[0] = absurd.iloc[0] * 50 + 1000
    assert guard.is_outlier(absurd)[0]
    assert guard.novelty(absurd)[0] > 1.0


def test_novelty_is_a_readable_multiple_of_the_threshold():
    guard = MahalanobisGuard(quantile=0.98).fit(STORE[COLS])
    nov = guard.novelty(STORE[COLS])
    flagged = guard.is_outlier(STORE[COLS])
    assert (nov[flagged] > 1.0).all()
    assert (nov[~flagged] <= 1.0).all()


def test_guard_survives_missing_and_infinite_values():
    guard = MahalanobisGuard(quantile=1.0).fit(STORE[COLS])
    bad = STORE[COLS].iloc[[0]].copy()
    bad.iloc[0, 0] = np.nan
    bad.iloc[0, 1] = np.inf
    assert np.isfinite(guard.distance(bad)).all()
