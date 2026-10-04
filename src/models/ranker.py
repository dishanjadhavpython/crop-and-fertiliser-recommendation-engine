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


#: Agronomic-fit columns where a higher value can only mean a better-suited
#: crop. Constraining them says so. Without it the ranker is free to learn,
#: from 34 districts, that a crop which fits its own envelope *better* should
#: rank lower — which is never agronomy, only noise fitted with confidence.
MONOTONE_UP = ("fit_min", "fit_mean", "fit_rain", "fit_temp", "fit_pH",
               "fit_depth", "fit_drainage", "fit_salinity", "fit_LGP", "fit_texture")


def monotone_constraints(features: list[str]) -> list[int]:
    return [1 if f in MONOTONE_UP else 0 for f in features]


def params(features: list[str], monotone: bool = True, **overrides) -> dict:
    """The ranker's parameters — one definition, shared by serving and evaluation.

    Anything that changes here changes both, which is the point: a benchmark
    that measures a differently-configured model is measuring the wrong thing.
    """
    p = dict(RANKER_PARAMS)
    if monotone:
        p["monotone_constraints"] = monotone_constraints(features)
    p.update(overrides)
    return p


class BaggedRanker:
    """Several seeds of the same ranker, averaged.

    A single LambdaMART fit on 34 districts is noticeably seed-dependent; the
    evaluation bagged that away and serving did not.
    """

    def __init__(self, models):
        self.models = list(models)

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)

    def __len__(self):
        return len(self.models)


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


def chosen_backend(stage: str = "ranker") -> str:
    """Which model family the tournament selected for this stage.

    Read from `artifacts/model_selection.json` rather than hardcoded. A
    tournament whose result is recorded and then ignored is worse than no
    tournament at all: it looks like evidence and changes nothing, which is
    exactly the trap this project fell into once before, when the benchmark
    bagged three seeds and serving shipped one.

    `pipeline.cache_inputs` hashes that file, so a change of model can never be
    served out of a stale fit. A missing or malformed artifact falls back to the
    incumbent — silently changing which model serves because a JSON file went
    bad is the one outcome worth ruling out.
    """
    import json

    path = config.ARTIFACTS / "model_selection.json"
    if not path.exists():
        return "lightgbm"
    try:
        return str(json.loads(path.read_text())["chosen"][stage])
    except Exception:                     # noqa: BLE001 - fall back, never guess
        return "lightgbm"


def fit(df: pd.DataFrame, features: list[str], seed: int = config.SEED,
        seeds: int = 1, monotone: bool = True, backend: str | None = None, **overrides):
    """Train the selected ranker on the full frame, averaged over ``seeds``."""
    backend = backend or chosen_backend("ranker")
    idx, groups = _grouped(df, np.arange(len(df)))

    if backend == "catboost":
        from src.models.backends import CatBoostRankerBackend

        return CatBoostRankerBackend(seeds=seeds, monotone=monotone, seed=seed,
                                     **overrides).fit(
            df.iloc[idx][features], df.iloc[idx]["relevance"], groups)
    if backend != "lightgbm":
        raise ValueError(f"no serving path for ranker backend {backend!r}; "
                         f"the tournament may record it, but nothing can fit it here")

    p = params(features, monotone=monotone, **overrides)
    models = []
    for s in range(seeds):
        model = LGBMRanker(**{**p, "random_state": seed + s})
        model.fit(df.iloc[idx][features], df.iloc[idx]["relevance"], group=groups)
        models.append(model)
    return models[0] if seeds == 1 else BaggedRanker(models)


def cross_val_scores(
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = config.N_FOLDS,
    seeds: int = config.N_SEEDS,
    use_weights: bool = False,
    monotone: bool = True,
    backend: str | None = None,
    **overrides,
) -> pd.Series:
    """Out-of-fold ranking scores under GroupKFold by district.

    Averaged over ``seeds`` random seeds: §7.2 shows seed bagging moving R2
    from 0.227 to 0.233 — small, free, and it reduces variance.

    These scores set the blend's alpha weights, so they **must** come from the
    same family that serves. Fitting alphas on LightGBM while CatBoost answers
    the request would weight one model by another's skill, which is the same
    class of mistake as benchmarking a bagged ranker and shipping a single fit.
    """
    backend = backend or chosen_backend("ranker")
    if backend == "catboost" and use_weights:
        raise ValueError("the CatBoost ranking backend takes no sample weights; "
                         "call with use_weights=False or keep LightGBM")
    params_ = params(features, monotone=monotone)

    acc = np.zeros(len(df))
    for s in range(seeds):
        fold_scores = np.full(len(df), np.nan)
        for train_idx, test_idx in group_kfold(df, n_splits):
            tr, groups = _grouped(df, train_idx)
            if backend == "catboost":
                from src.models.backends import CatBoostRankerBackend

                model = CatBoostRankerBackend(seeds=1, monotone=monotone,
                                              seed=config.SEED + s, **overrides)
                model.fit(df.iloc[tr][features], df.iloc[tr]["relevance"], groups)
                fold_scores[test_idx] = model.predict(df.iloc[test_idx][features])
                continue
            model = LGBMRanker(**{**params_, "random_state": config.SEED + s, **overrides})
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


def feature_importance(model, features: list[str], top: int = 25) -> pd.DataFrame:
    """Gain per feature, from whichever family actually fitted the model.

    The served ranker is no longer necessarily LightGBM, so reaching straight
    into `booster_` raises on a CatBoost fit. Handled here rather than left to
    the caller because this is the sort of helper someone reaches for from a
    notebook months later, where an AttributeError reads as a puzzle instead of
    a signal.
    """
    inner = getattr(model, "models", None)
    first = inner[0] if inner else model            # bagged, or a backend wrapper
    gain = (first.booster_.feature_importance("gain") if hasattr(first, "booster_")
            else first.get_feature_importance())
    imp = pd.DataFrame({"feature": features, "gain": gain})
    return imp.sort_values("gain", ascending=False).head(top).reset_index(drop=True)
