"""Evaluation metrics (plan §7.5).

Ranking metrics are primary. Yield is reported as within-crop Spearman and
per-crop MAE — never as pooled R2, which is dominated by the sugarcane scale
effect and produces the fake 0.928 the plan opens with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ------------------------------------------------------------- ranking ----
def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(((2 ** rel - 1) / discounts).sum())


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5) -> float:
    """Normalised DCG for one query."""
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")
    ideal = dcg_at_k(np.sort(y_true)[::-1], k)
    return dcg_at_k(y_true[order], k) / ideal if ideal > 0 else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 3,
                   relevant_from: int = 3) -> float:
    """Share of the top k that is genuinely relevant (grade >= ``relevant_from``)."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")[:k]
    return float((y_true[order] >= relevant_from).mean()) if len(order) else 0.0


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 5,
                relevant_from: int = 3) -> float:
    y_true = np.asarray(y_true)
    n_relevant = int((y_true >= relevant_from).sum())
    if n_relevant == 0:
        return np.nan
    order = np.argsort(-np.asarray(y_score, dtype=float), kind="stable")[:k]
    return float((y_true[order] >= relevant_from).sum() / n_relevant)


def query_columns(df: pd.DataFrame) -> list[str]:
    """The columns that define one ranking query.

    Year is part of the query whenever the frame is a panel. Omitting it pools
    every year into a single enormous query, which silently wrecks the
    recall metric — with eight years of relevant crops competing for five
    slots, Recall@5 is capped near 0.2 no matter how good the ranking is.
    """
    cols = ["District", "Season"]
    if "Year" in df.columns and df["Year"].nunique() > 1:
        cols.append("Year")
    return cols


def ranking_report(df: pd.DataFrame, score_col: str, truth_col: str = "relevance",
                   group_cols=None) -> dict:
    """NDCG@5, NDCG@3, P@3 and Recall@5, averaged over query groups."""
    group_cols = list(group_cols) if group_cols is not None else query_columns(df)
    rows = []
    for _, g in df.groupby(list(group_cols)):
        if len(g) < 2:
            continue
        t, s = g[truth_col].to_numpy(), g[score_col].to_numpy()
        rows.append({
            "ndcg@5": ndcg_at_k(t, s, 5),
            "ndcg@3": ndcg_at_k(t, s, 3),
            "p@3": precision_at_k(t, s, 3),
            "recall@5": recall_at_k(t, s, 5),
        })
    out = pd.DataFrame(rows)
    return {k: round(float(np.nanmean(out[k])), 3) for k in out.columns}


# --------------------------------------------------------------- yield ----
def within_crop_spearman(df: pd.DataFrame, pred_col: str, truth_col: str = "yield_z") -> float:
    """Mean Spearman rho computed inside each crop.

    Pooled correlation would mostly measure the crop-scale effect; this
    measures whether the model orders *districts* correctly for a given crop,
    which is the question the recommender actually asks.
    """
    rhos = []
    for _, g in df.groupby("Crop"):
        if len(g) < 4 or g[pred_col].nunique() < 2:
            continue
        rho = spearmanr(g[truth_col], g[pred_col]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
    return round(float(np.mean(rhos)), 3) if rhos else float("nan")


def per_crop_spearman(df: pd.DataFrame, pred_col: str,
                      truth_col: str = "yield_z") -> pd.DataFrame:
    """Per-crop rho — the §7.4 table, and the source of the S5 blend weights."""
    rows = []
    for crop, g in df.groupby("Crop"):
        if len(g) < 4 or g[pred_col].nunique() < 2:
            rows.append({"Crop": crop, "n": len(g), "rho": np.nan, "mae_t_ha": np.nan})
            continue
        rho = spearmanr(g[truth_col], g[pred_col]).statistic
        mae = np.nan
        if {"crop_yield_mean", "crop_yield_std", "Yield"} <= set(g.columns):
            pred_t = g[pred_col] * g["crop_yield_std"] + g["crop_yield_mean"]
            mae = float((pred_t - g["Yield"]).abs().mean())
        rows.append({"Crop": crop, "n": len(g), "rho": round(float(rho), 3),
                     "mae_t_ha": round(mae, 3) if np.isfinite(mae) else np.nan})
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else float("nan")


# --------------------------------------------------------- calibration ----
def interval_coverage(y_true, lower, upper) -> dict:
    """Empirical coverage and mean width of a prediction interval."""
    y_true = np.asarray(y_true, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    inside = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": round(float(inside.mean()), 3),
        "mean_width": round(float((upper - lower).mean()), 3),
        "n": int(len(y_true)),
    }
