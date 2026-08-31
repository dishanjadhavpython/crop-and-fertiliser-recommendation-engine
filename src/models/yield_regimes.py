"""S3 — two yield regimes, because they are genuinely different products.

A district with cropping history and a district without are not the same
prediction problem, and collapsing them into one number misreports both.

    warm start   history exists -> panel-history features are available
    cold start   no history     -> soil, climate and crop identity only

Measured under the protocol that matches each regime:

    cold start, GroupKFold by district, no lags      rho 0.420
    warm start, forward chaining, with lags          rho 0.506

The temptation is to quote the GroupKFold-with-lags number (rho 0.496, R2
0.309) because it is the largest. It is also the least honest: under
GroupKFold a held-out district's lags come from its own held-out rows, so the
model is handed a history that a genuinely new district does not have. It is
reported here as an upper bound and never as the headline.

Serving picks the regime from the query, not from what scores better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import config
from src.eval.metrics import per_crop_spearman, r2, within_crop_spearman
from src.eval.splits import forward_chaining, group_kfold
from src.features.panel_history import LAG_COLUMNS, add as add_history
from src.models import yield_quantile

#: per-crop cross-validated rho below which no yield number is served at all
SKILL_FLOOR = 0.25


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Training rows with panel history attached."""
    return add_history(yield_quantile.training_rows(df))


def feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(cold-start features, warm-start features)."""
    from src.data.training_set import model_features

    all_feats = model_features(df)
    cold = [c for c in all_feats if c not in LAG_COLUMNS]
    warm = cold + [c for c in LAG_COLUMNS if c in df.columns]
    return cold, warm


def _oof(df: pd.DataFrame, feats: list[str], splitter, seeds: int = 3,
         target: str = "yield_z") -> np.ndarray:
    acc = np.zeros(len(df))
    cnt = np.zeros(len(df))
    for s in range(seeds):
        for tr, te in splitter(df):
            m = LGBMRegressor(
                **{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
                random_state=config.SEED + s)
            m.fit(df.iloc[tr][feats], df.iloc[tr][target])
            acc[te] += m.predict(df.iloc[te][feats])
            cnt[te] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def evaluate(df: pd.DataFrame, seeds: int = 3) -> pd.DataFrame:
    """The regime comparison table, each regime under its matching protocol."""
    d = prepare(df)
    cold, warm = feature_sets(d)

    rows = []
    for label, feats, splitter, note in [
        ("cold start (new district, no history)", cold, group_kfold,
         "GroupKFold by district — the honest number for an unseen place"),
        ("warm start (known district, next year)", warm, forward_chaining,
         "forward chaining — the honest number for a place with history"),
        ("upper bound (do not report as headline)", warm, group_kfold,
         "GroupKFold with lags: a held-out district's history comes from its "
         "own held-out rows, so this overstates the cold-start case"),
    ]:
        d = d.copy()
        d["_p"] = _oof(d, feats, splitter, seeds)
        m = d.dropna(subset=["_p"])
        rows.append({
            "regime": label, "n": len(m), "n_features": len(feats),
            "R2": r2(m["yield_z"], m["_p"]),
            "within_crop_rho": within_crop_spearman(m, "_p"),
            "note": note,
        })
    return pd.DataFrame(rows)


def crop_skill(df: pd.DataFrame, seeds: int = 3) -> pd.DataFrame:
    """Per-crop skill under each regime — drives the abstention rule."""
    d = prepare(df)
    cold, warm = feature_sets(d)
    out = None
    for label, feats, splitter in [("cold", cold, group_kfold),
                                   ("warm", warm, forward_chaining)]:
        dd = d.copy()
        dd["_p"] = _oof(dd, feats, splitter, seeds)
        s = per_crop_spearman(dd.dropna(subset=["_p"]), "_p")[["Crop", "n", "rho"]]
        s = s.rename(columns={"rho": f"rho_{label}", "n": f"n_{label}"})
        out = s if out is None else out.merge(s, on="Crop", how="outer")
    # Abstention is decided PER REGIME, not globally. Taking the max across
    # regimes would serve a crop in both whenever either is skilled — and the
    # regimes answer different queries. Safflower is the case that exposed it:
    # rho_warm 0.263 but rho_cold -0.130, worse than predicting the mean. A
    # cold-start query for safflower must get no number at all.
    out["serve_cold"] = out["rho_cold"].fillna(-1) >= SKILL_FLOOR
    out["serve_warm"] = out["rho_warm"].fillna(-1) >= SKILL_FLOOR
    out["serve_yield_number"] = out["serve_cold"] | out["serve_warm"]
    return out.sort_values("rho_warm", ascending=False).reset_index(drop=True)


def fit(df: pd.DataFrame) -> dict:
    """Fit both regimes plus their quantile bands, for serving."""
    d = prepare(df)
    cold, warm = feature_sets(d)
    warm_rows = d[d["has_history"] == 1]

    models = {"cold_features": cold, "warm_features": warm}
    for name, feats, rows in [("cold", cold, d), ("warm", warm, warm_rows)]:
        models[f"{name}_point"] = LGBMRegressor(
            **{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
            random_state=config.SEED).fit(rows[feats], rows["yield_z"])
        models[f"{name}_quantiles"] = {
            a: LGBMRegressor(**yield_quantile.QUANTILE_PARAMS, alpha=a,
                             random_state=config.SEED).fit(rows[feats], rows["yield_z"])
            for a in config.QUANTILES
        }
    return models
