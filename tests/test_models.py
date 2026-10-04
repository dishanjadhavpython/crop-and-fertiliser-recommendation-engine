"""Learned components and the calibration layer (plan §5, §7)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.training_set import build, model_features
from src.eval import baselines
from src.eval.metrics import (
    dcg_at_k,
    interval_coverage,
    ndcg_at_k,
    per_crop_spearman,
    precision_at_k,
    r2,
    ranking_report,
    recall_at_k,
)
from src.models import blend, conformal, ranker, yield_quantile

warnings.filterwarnings("ignore")

DF = build()
FEATS = model_features(DF)
Y = yield_quantile.training_rows(DF)


# --------------------------------------------------------------- metrics ----
def test_ndcg_bounds():
    t = np.array([4, 3, 2, 1, 0])
    assert ndcg_at_k(t, t, 5) == pytest.approx(1.0)
    assert 0 <= ndcg_at_k(t, -t, 5) < 1
    assert ndcg_at_k(np.zeros(5), np.arange(5), 5) == 0.0


def test_dcg_discounts_later_positions():
    assert dcg_at_k([3, 0], 2) > dcg_at_k([0, 3], 2)


def test_precision_and_recall_at_k():
    t = np.array([4, 0, 3, 0, 0])
    s = np.array([5, 4, 3, 2, 1])
    assert precision_at_k(t, s, 3) == pytest.approx(2 / 3)
    assert recall_at_k(t, s, 5) == pytest.approx(1.0)
    assert np.isnan(recall_at_k(np.zeros(5), s, 5))


def test_r2_of_a_perfect_and_a_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0
    assert r2(y, np.full(4, y.mean())) == 0.0


# ------------------------------------------------------------- baselines ----
def test_popularity_prior_beats_random_by_a_wide_margin():
    df = DF.copy()
    df["r"] = baselines.random_baseline(df)
    df["p"] = baselines.popularity_prior_oof(df)
    rand = ranking_report(df, "r")["ndcg@5"]
    pop = ranking_report(df, "p")["ndcg@5"]
    assert rand < 0.30
    assert pop > 0.70
    assert pop - rand > 0.4


def test_rule_baseline_needs_no_labels_and_still_beats_random():
    df = DF.copy()
    df["rules"] = baselines.rule_score_baseline(df)
    df["r"] = baselines.random_baseline(df)
    assert ranking_report(df, "rules")["ndcg@5"] > ranking_report(df, "r")["ndcg@5"]


# ---------------------------------------------------------------- ranker ----
def test_ranker_clears_the_popularity_bar():
    """Plan §7.3: the number that must be reported and beaten."""
    df = DF.copy()
    df["pop"] = baselines.popularity_prior_oof(df)
    df["model"] = ranker.cross_val_scores(df, FEATS, seeds=2)
    pop = ranking_report(df, "pop")
    model = ranking_report(df, "model")
    assert model["ndcg@5"] > pop["ndcg@5"]
    assert model["ndcg@5"] > 0.80


def test_engineered_features_beat_crop_identity_alone():
    df = DF.copy()
    df["ident"] = ranker.cross_val_scores(df, ["crop_id", "season_id"], seeds=2)
    df["full"] = ranker.cross_val_scores(df, FEATS, seeds=2)
    gain = ranking_report(df, "full")["ndcg@5"] - ranking_report(df, "ident")["ndcg@5"]
    assert gain > 0.1


# --------------------------------------------------------- yield quantile ---
def test_raw_yield_target_is_the_documented_mirage():
    """Predicting raw yield scores impressively while learning nothing about agronomy.

    The absolute number has moved as the panel changed — 0.93 on one year of
    labels, 0.82 on eight, 0.75 once climate features became as-of normals — so
    it is not what this test pins. The *mirage* is: a model given only the crop
    name matches the full feature set, and now beats it (0.787 against 0.750).
    What looks like agronomy is crop-scale arithmetic, sugarcane at 74 t/ha
    against sesamum at 0.27.
    """
    from lightgbm import LGBMRegressor
    from src.eval.splits import group_kfold

    def oof(feats, target):
        pred = np.full(len(Y), np.nan)
        for tr, te in group_kfold(Y):
            m = LGBMRegressor(**{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
                              random_state=config.SEED)
            m.fit(Y.iloc[tr][feats], Y.iloc[tr][target])
            pred[te] = m.predict(Y.iloc[te][feats])
        return pred

    full = r2(Y["Yield"], oof(FEATS, "Yield"))
    ident = r2(Y["Yield"], oof(["crop_id", "season_id"], "Yield"))
    honest = r2(Y["yield_z"], oof(FEATS, "yield_z"))

    assert full > 0.60                     # still an impressive-looking number
    # ... and a model given ONLY the crop name gets all of it. That gap is the
    # whole point: it is what the soil and climate features are really worth on
    # the raw target, and it is nothing.
    assert full - ident < 0.10
    assert 0.10 < honest < 0.40            # the real signal


def test_quantiles_are_ordered_and_cover_sensibly():
    q = yield_quantile.cross_val_quantiles(Y, FEATS, seeds=1)
    assert (q["p10"] <= q["p50"]).all()
    assert (q["p50"] <= q["p90"]).all()
    cov = interval_coverage(Y["yield_z"], q["p10"], q["p90"])
    assert 0.5 < cov["coverage"] < 0.95    # under-covered before conformal widening


def test_conversion_back_to_tonnes_is_the_inverse_of_the_z_score():
    z, mean, std = np.array([0.0, 1.0, -1.0]), 2.5, 0.4
    back = yield_quantile.to_tonnes_per_ha(z, mean, std)
    assert back[0] == pytest.approx(2.5)
    assert back[1] == pytest.approx(2.9)
    assert back[2] == pytest.approx(2.1)


# ------------------------------------------------------------- conformal ----
def test_conformal_widening_restores_nominal_coverage():
    """A 90% interval that covers ~90% of held-out districts is a checkable claim."""
    y = Y.join(yield_quantile.cross_val_quantiles(Y, FEATS, seeds=1))
    raw = interval_coverage(y["yield_z"], y["p10"], y["p90"])["coverage"]
    c = conformal.cross_val_conformal(y, "p10", "p90", alpha=0.1)
    y = y.join(c)
    got = interval_coverage(y["yield_z"], y["conf_lo_90"], y["conf_hi_90"])
    assert got["coverage"] >= 0.85
    assert got["coverage"] > raw
    assert got["mean_width"] > (y["p90"] - y["p10"]).mean()


def test_conformal_intervals_widen_as_alpha_shrinks():
    y = Y.join(yield_quantile.cross_val_quantiles(Y, FEATS, seeds=1))
    widths = {}
    for alpha in (0.2, 0.1):
        c = conformal.cross_val_conformal(y, "p10", "p90", alpha=alpha)
        widths[alpha] = float((c.iloc[:, 1] - c.iloc[:, 0]).mean())
    assert widths[0.1] > widths[0.2]


def test_mahalanobis_guard_flags_a_synthetic_outlier():
    rng = np.random.default_rng(config.SEED)
    X = rng.normal(size=(40, 12))
    g = conformal.MahalanobisGuard().fit(X)
    assert g.is_outlier(X).mean() < 0.15
    assert g.is_outlier(np.full((1, 12), 15.0))[0]


# ----------------------------------------------------------------- blend ----
def test_alpha_is_zero_where_the_model_is_worse_than_the_mean():
    skill = pd.DataFrame({
        "Crop": ["good", "middling", "bad", "useless"],
        "n": [30, 30, 30, 30],
        "rho": [0.80, 0.40, 0.10, -0.30],
        "mae_t_ha": [0.1] * 4,
    })
    a = blend.alpha_from_skill(skill)
    assert a["good"] == 1.0
    assert 0 < a["middling"] < 1
    assert a["bad"] == 0.0
    assert a["useless"] == 0.0


def test_blend_falls_back_to_rules_when_out_of_distribution():
    df = pd.DataFrame({
        "District": ["A"] * 4, "Season": ["Kharif"] * 4,
        "Crop": ["c1", "c2", "c3", "c4"],
        "learned": [4.0, 3.0, 2.0, 1.0],
        "rules": [1.0, 2.0, 3.0, 4.0],
    })
    alphas = {c: 1.0 for c in df["Crop"]}
    trusted = blend.blend(df, "learned", "rules", alphas)
    ood = blend.blend(df, "learned", "rules", alphas, ood_mask=np.ones(4, dtype=bool))
    assert trusted.idxmax() == 0        # model's favourite wins
    assert ood.idxmax() == 3            # rules' favourite wins instead


def test_rank_norm_is_bounded_and_order_preserving():
    s = pd.Series([5.0, 1.0, 3.0])
    r = blend.rank_norm(s)
    assert r.between(0, 1).all()
    assert r.idxmax() == 0 and r.idxmin() == 1
    assert blend.rank_norm(pd.Series([2.0, 2.0])).eq(0.5).all()


def test_more_label_years_repaired_the_crops_the_plan_called_broken():
    """Cotton and gram were worse than the mean on one year of labels.

    The plan attributed that to the missing irrigation variable (§7.4). Eight
    years of labels repaired both without adding any new variable: cotton moved
    from rho = -0.33 to positive, gram from -0.28 to positive. That is the
    clearest single piece of evidence that n=34 — not the feature set — was the
    binding constraint.
    """
    from lightgbm import LGBMRegressor
    from src.eval.splits import group_kfold

    y = Y.copy()
    pred = np.full(len(y), np.nan)
    for tr, te in group_kfold(y):
        m = LGBMRegressor(**{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
                          random_state=config.SEED)
        m.fit(y.iloc[tr][FEATS], y.iloc[tr]["yield_z"])
        pred[te] = m.predict(y.iloc[te][FEATS])
    y["pred"] = pred
    skill = per_crop_spearman(y, "pred").set_index("Crop")["rho"]
    assert skill["Cotton(lint)"] > 0.1
    assert skill["Gram"] > 0.1
    assert skill.max() > 0.6
    # and the count of crops the model cannot order at all has collapsed
    assert (skill < 0.2).sum() <= 4
