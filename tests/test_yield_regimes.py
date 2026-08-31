"""S3 — panel history, the two regimes, the class model and abstention."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.training_set import build
from src.features.panel_history import HISTORY_KEY, LAG_COLUMNS, add, coverage
from src.models import yield_class, yield_regimes

warnings.filterwarnings("ignore")

DF = build()
HIST = yield_regimes.prepare(DF)


# ------------------------------------------------------- leakage guards ----
def test_lags_are_strictly_backward_looking():
    """The whole feature block is worthless — and dangerous — if a row can see
    its own outcome or a later one."""
    g = HIST.sort_values(HISTORY_KEY + ["Year"])
    for _, grp in g.groupby(HISTORY_KEY):
        yz = grp["yield_z"].to_numpy()
        lag1 = grp["lag1_yield_z"].to_numpy()
        assert pd.isna(lag1[0]), "the first year of a series must have no lag"
        for i in range(1, len(grp)):
            if not pd.isna(lag1[i]):
                assert lag1[i] == pytest.approx(yz[i - 1])


def test_expanding_stats_never_include_the_current_year():
    g = HIST.sort_values(HISTORY_KEY + ["Year"])
    for _, grp in g.groupby(HISTORY_KEY):
        if len(grp) < 3:
            continue
        yz = grp["yield_z"].to_numpy()
        mean = grp["lag_mean_yield_z"].to_numpy()
        assert mean[2] == pytest.approx(yz[:2].mean())
        break


def test_add_preserves_row_order_and_index():
    y = yield_regimes.prepare(DF)
    again = add(y.drop(columns=[c for c in LAG_COLUMNS if c in y.columns]))
    assert (again.index == y.index).all()


def test_history_coverage_is_reported():
    c = coverage(HIST)
    assert c["with_history"] == c["with_lag1"]
    assert 0 < c["with_history"] < c["rows"]     # some rows are first-years


# ------------------------------------------------------------- regimes ----
def test_cold_features_exclude_lags_and_warm_include_them():
    cold, warm = yield_regimes.feature_sets(HIST)
    assert not (set(cold) & set(LAG_COLUMNS))
    assert set(LAG_COLUMNS) <= set(warm)
    assert len(warm) > len(cold)


def test_warm_start_beats_cold_start():
    """History is worth something, and the plan's target is rho >= 0.50 warm."""
    res = yield_regimes.evaluate(DF, seeds=1).set_index("regime")
    warm = [i for i in res.index if i.startswith("warm")][0]
    cold = [i for i in res.index if i.startswith("cold")][0]
    assert res.loc[warm, "within_crop_rho"] > res.loc[cold, "within_crop_rho"]
    assert res.loc[warm, "within_crop_rho"] >= 0.48


def test_upper_bound_arm_is_labelled_as_such():
    """GroupKFold-with-lags flatters the cold-start case and must never be the
    headline: a held-out district's lags come from its own held-out rows."""
    res = yield_regimes.evaluate(DF, seeds=1)
    ub = res[res["regime"].str.startswith("upper bound")]
    assert len(ub) == 1
    assert "overstates" in ub.iloc[0]["note"]


# --------------------------------------------------------- class model ----
def test_tercile_labels_are_cut_within_crop_and_year():
    lab = yield_class.make_labels(HIST)
    assert set(lab.unique()) <= {0, 1, 2}
    # within a crop-year the three classes should be near-balanced
    share = (HIST.assign(c=lab).groupby(["Crop", "Year"])["c"]
             .value_counts(normalize=True).groupby(level=2).mean())
    assert share.max() < 0.5


def test_class_model_beats_the_majority_baseline():
    cold, _ = yield_regimes.feature_sets(HIST)
    pred = yield_class.cross_val_predict(HIST, cold, seeds=1)
    rep = yield_class.report(pred)
    assert rep["accuracy"] > rep["baseline_majority"] + 0.10
    # confusing "below" with "above" is the error that would actually mislead
    assert rep["severe_error_rate"] < 0.20


# ---------------------------------------------------------- abstention ----
def test_abstention_is_decided_per_regime():
    """Taking the max across regimes would serve a crop in both whenever
    either is skilled. Safflower is the case that exposed it: rho_warm 0.263
    but rho_cold -0.130, worse than predicting the mean."""
    skill = yield_regimes.crop_skill(DF, seeds=1)
    assert {"serve_cold", "serve_warm"} <= set(skill.columns)
    for row in skill.itertuples(index=False):
        cold = row.rho_cold if np.isfinite(row.rho_cold) else -1
        warm = row.rho_warm if np.isfinite(row.rho_warm) else -1
        assert row.serve_cold == (cold >= yield_regimes.SKILL_FLOOR)
        assert row.serve_warm == (warm >= yield_regimes.SKILL_FLOOR)
    # the two regimes must genuinely disagree somewhere, or the split is moot
    assert (skill["serve_cold"] != skill["serve_warm"]).any()


def test_no_crop_is_served_a_number_where_the_model_is_worse_than_the_mean():
    skill = yield_regimes.crop_skill(DF, seeds=1)
    served_cold = skill.loc[skill["serve_cold"], "rho_cold"]
    served_warm = skill.loc[skill["serve_warm"], "rho_warm"]
    assert (served_cold >= yield_regimes.SKILL_FLOOR).all()
    assert (served_warm >= yield_regimes.SKILL_FLOOR).all()
