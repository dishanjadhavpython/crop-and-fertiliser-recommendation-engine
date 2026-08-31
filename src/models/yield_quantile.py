"""S3 — yield and confidence (plan §5.3). Learned.

Given an honest R2 around 0.23, a point yield estimate would be misleading.
Three LightGBM models are trained with ``objective="quantile"`` at alpha =
0.1, 0.5, 0.9 and the output is a band — *"typically 1.8-2.9 t/ha in districts
like yours"* — which is both more honest and more useful than a spurious 2.34.

The target is the **within-crop z-score** of yield, never raw yield. Sugarcane
at 74 t/ha against sesamum at 0.27 t/ha would otherwise dominate every split in
the tree and produce the fake R2 = 0.928 the plan opens with. Predictions are
converted back to t/ha per crop at serving time using that crop's own mean and
standard deviation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import config
from src.eval.splits import group_kfold

QUANTILE_PARAMS = dict(
    objective="quantile",
    num_leaves=15,
    min_child_samples=10,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    verbose=-1,
)


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Only district-crop pairs with an actual recorded yield."""
    return df[df["has_yield"]].reset_index(drop=True)


def _weights(df: pd.DataFrame) -> np.ndarray:
    """Sample weight from SHC coverage.

    A district whose talukas were sampled 4,480 times deserves more trust than
    one sampled 200 times — most pipelines throw this information away.
    """
    w = np.log1p(df["shc_samples_total"].to_numpy(dtype=float))
    return w / w.mean()


def fit_quantiles(df: pd.DataFrame, features: list[str], seed: int = config.SEED
                  ) -> dict[float, LGBMRegressor]:
    models = {}
    w = _weights(df)
    for alpha in config.QUANTILES:
        m = LGBMRegressor(**QUANTILE_PARAMS, alpha=alpha, random_state=seed)
        m.fit(df[features], df["yield_z"], sample_weight=w)
        models[alpha] = m
    return models


def cross_val_quantiles(
    df: pd.DataFrame,
    features: list[str],
    n_splits: int = config.N_FOLDS,
    seeds: int = config.N_SEEDS,
    select_top: int | None = None,
) -> pd.DataFrame:
    """Out-of-fold p10/p50/p90 predictions under GroupKFold by district.

    ``select_top`` runs nested importance-based feature selection **inside each
    fold**. Selecting on the full dataset and then cross-validating is a leak,
    and a common one (plan §4.4).
    """
    out = {a: np.zeros(len(df)) for a in config.QUANTILES}

    for s in range(seeds):
        fold = {a: np.full(len(df), np.nan) for a in config.QUANTILES}
        for train_idx, test_idx in group_kfold(df, n_splits):
            tr, te = df.iloc[train_idx], df.iloc[test_idx]
            feats = features
            if select_top:
                feats = select_features(tr, features, select_top, seed=config.SEED + s)
            w = _weights(tr)
            for alpha in config.QUANTILES:
                m = LGBMRegressor(**QUANTILE_PARAMS, alpha=alpha, random_state=config.SEED + s)
                m.fit(tr[feats], tr["yield_z"], sample_weight=w)
                fold[alpha][test_idx] = m.predict(te[feats])
        for a in config.QUANTILES:
            out[a] += fold[a]

    res = pd.DataFrame({f"p{int(a*100)}": out[a] / seeds for a in config.QUANTILES},
                       index=df.index)
    # quantile models are fitted independently and can cross; sorting each row
    # restores monotonicity without changing the marginal calibration much
    res[["p10", "p50", "p90"]] = np.sort(res[["p10", "p50", "p90"]].to_numpy(), axis=1)
    return res


def select_features(train: pd.DataFrame, features: list[str], k: int,
                    seed: int = config.SEED) -> list[str]:
    """Importance-based selection fitted on the training fold only."""
    m = LGBMRegressor(**{**QUANTILE_PARAMS, "objective": "regression"}, random_state=seed)
    m.fit(train[features], train["yield_z"], sample_weight=_weights(train))
    gain = pd.Series(m.booster_.feature_importance("gain"), index=features)
    return gain.sort_values(ascending=False).head(k).index.tolist()


def to_tonnes_per_ha(z: np.ndarray | pd.Series, crop_mean, crop_std):
    """Convert a within-crop z-score prediction back to t/ha."""
    return np.asarray(z, dtype=float) * np.asarray(crop_std, dtype=float) \
        + np.asarray(crop_mean, dtype=float)
