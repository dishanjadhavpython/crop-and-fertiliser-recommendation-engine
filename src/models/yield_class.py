"""S3b — yield *class* instead of a yield number (review item 4).

A conformal band of 0.4-1.6 t/ha around a median of 1.0 is honest and
useless: it does not change any decision a farmer would make. The data
supports a coarser question far better than a precise one, so this predicts
where a district-crop sits relative to that crop's own norm:

    below   — bottom tercile of districts for this crop
    typical — middle tercile
    above   — top tercile

Terciles are cut **within crop and year**, so the classes mean the same thing
for sugarcane as for sesamum, and a good year does not push every district
into "above". A three-class problem at within-crop rho ~0.43 is answerable;
a point estimate at R2 ~0.19 is not.

The quantile band in ``yield_quantile`` is kept alongside this — it remains the
right output where the model is genuinely skilled on a crop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src import config
from src.eval.splits import group_kfold

CLASSES = ["below", "typical", "above"]

PARAMS = dict(
    objective="multiclass",
    num_class=3,
    num_leaves=15,
    min_child_samples=10,
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    verbose=-1,
)


def make_labels(df: pd.DataFrame) -> pd.Series:
    """Tercile of yield within (crop, year) — 0 below, 1 typical, 2 above."""
    def tercile(s: pd.Series) -> pd.Series:
        if s.nunique() < 3:
            return pd.Series(np.ones(len(s), dtype=int), index=s.index)
        try:
            return pd.qcut(s.rank(method="first"), 3, labels=False).astype(int)
        except ValueError:
            return pd.Series(np.ones(len(s), dtype=int), index=s.index)

    return df.groupby(["Crop", "Year"])["Yield"].transform(tercile).astype(int)


def fit(df: pd.DataFrame, features: list[str], seed: int = config.SEED) -> LGBMClassifier:
    m = LGBMClassifier(**PARAMS, random_state=seed)
    m.fit(df[features], make_labels(df))
    return m


def cross_val_predict(df: pd.DataFrame, features: list[str],
                      splitter=group_kfold, seeds: int = 3) -> pd.DataFrame:
    """Out-of-fold class probabilities under the given protocol."""
    y = make_labels(df).to_numpy()
    proba = np.zeros((len(df), 3))
    count = np.zeros(len(df))
    for s in range(seeds):
        for tr, te in splitter(df):
            m = LGBMClassifier(**PARAMS, random_state=config.SEED + s)
            m.fit(df.iloc[tr][features], y[tr])
            proba[te] += m.predict_proba(df.iloc[te][features])
            count[te] += 1
    keep = count > 0
    proba[keep] /= count[keep, None]
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in CLASSES], index=df.index)
    out["pred_class"] = np.where(keep, np.argmax(proba, axis=1), np.nan)
    out["true_class"] = y
    out["evaluated"] = keep
    return out


def report(pred: pd.DataFrame) -> dict:
    """Accuracy, plus the two figures that decide whether this is usable."""
    ev = pred[pred["evaluated"]]
    if ev.empty:
        return {}
    correct = ev["pred_class"] == ev["true_class"]
    # confusing "below" with "above" is the error that would actually mislead
    severe = (ev["pred_class"] - ev["true_class"]).abs() == 2
    conf = ev[[f"p_{c}" for c in CLASSES]].max(axis=1)
    high = conf >= 0.5
    return {
        "n": int(len(ev)),
        "accuracy": round(float(correct.mean()), 3),
        "baseline_majority": round(float(ev["true_class"].value_counts(normalize=True).max()), 3),
        "severe_error_rate": round(float(severe.mean()), 4),
        "coverage_at_conf50": round(float(high.mean()), 3),
        "accuracy_at_conf50": round(float(correct[high].mean()), 3) if high.any() else None,
    }
