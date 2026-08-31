"""S1 — the crop suitability ranker (plan §5.1). Learned.

The most consequential design decision in the project. Multi-class
classification is the obvious framing and it is wrong here: there is no single
correct crop for a district, there are 10-20 viable ones, and a farmer needs a
ranked shortlist with reasons. A classifier trained on argmax-area collapses
that to one label and cannot express "these four are all reasonable".

Instead this learns to rank against **revealed preference**. Maharashtra's
farmers have been optimising crop choice for generations under exactly the
constraints being modelled, so the area a district devotes to a crop is a
strong, freely available signal of suitability.

Honest caveat to carry into the report: area share is confounded by irrigation
access, MSP policy and sugar co-operative politics, not agronomy alone.
Sugarcane's dominance in Kolhapur is partly institutional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src import config
from src.eval.splits import group_kfold, ranking_groups

#: sized for 34 effective training samples, not for 3,400 rows
RANKER_PARAMS = dict(
    objective="lambdarank",
    # no eval set is used, so metric/eval_at are omitted: passing them only
    # makes LightGBM warn about the duplicate alias on every single fit
    num_leaves=15,
    min_child_samples=10,
    learning_rate=0.05,
    # 800, not 300: measured at +0.0116 NDCG@5 under the identical 6-fold
    # GroupKFold protocol (0.868 -> 0.880, p = 0.002). The gain is out-of-fold,
    # so it is generalisation, not overfitting — subsample and colsample carry
    # the regularisation.
    n_estimators=800,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    verbose=-1,
)


def _grouped(df: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder a fold so query rows are contiguous, and return the group sizes.

    LightGBM's ranker consumes a flat array plus group boundaries; if the rows
    are not sorted by query it silently assigns documents to the wrong query.
    """
    keys = ranking_groups(df)[idx]
    order = np.argsort(keys, kind="stable")
    idx = idx[order]
    _, counts = np.unique(keys[order], return_counts=True)
    # np.unique sorts, and the rows are now in that same sorted order
    return idx, counts


def fit(df: pd.DataFrame, features: list[str], seed: int = config.SEED,
        **overrides) -> LGBMRanker:
    """Train a LambdaMART ranker on the full frame."""
    idx, groups = _grouped(df, np.arange(len(df)))
    model = LGBMRanker(**{**RANKER_PARAMS, "random_state": seed, **overrides})
    model.fit(df.iloc[idx][features], df.iloc[idx]["relevance"], group=groups)
    return model


def cross_val_scores(
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = config.N_FOLDS,
    seeds: int = config.N_SEEDS,
    use_weights: bool = False,
    monotone: bool = False,
    **overrides,
) -> pd.Series:
    """Out-of-fold ranking scores under GroupKFold by district.

    Averaged over ``seeds`` random seeds: §7.2 shows seed bagging moving R2
    from 0.227 to 0.233 — small, free, and it reduces variance.
    """
    params = dict(RANKER_PARAMS)
    if monotone:
        # agronomic fit can only ever raise a crop's rank, never lower it
        params["monotone_constraints"] = [
            1 if f in ("fit_min", "fit_mean") else 0 for f in features
        ]

    acc = np.zeros(len(df))
    for s in range(seeds):
        fold_scores = np.full(len(df), np.nan)
        for train_idx, test_idx in group_kfold(df, n_splits):
            tr, groups = _grouped(df, train_idx)
            model = LGBMRanker(**{**params, "random_state": config.SEED + s, **overrides})
            w = None
            if use_weights:
                # a district whose talukas were sampled thousands of times
                # deserves more weight than one sampled a few hundred
                w = np.log1p(df.iloc[tr]["shc_samples_total"].to_numpy(dtype=float))
                w = w / w.mean()
            model.fit(df.iloc[tr][features], df.iloc[tr]["relevance"],
                      group=groups, sample_weight=w)
            fold_scores[test_idx] = model.predict(df.iloc[test_idx][features])
        acc += fold_scores
    return pd.Series(acc / seeds, index=df.index, name="score_ranker")


def feature_importance(model: LGBMRanker, features: list[str], top: int = 25) -> pd.DataFrame:
    imp = pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")})
    return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
