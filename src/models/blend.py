"""S5 — per-crop blending of the learned ranker and the rule scorer (plan §5.4).

    final = alpha_crop * rank_norm(S1) + (1 - alpha_crop) * rank_norm(S2)

Alpha is set **per crop** from cross-validated skill, because the model's skill
varies enormously by crop: within-crop Spearman runs from 0.82 for sesamum down
to negative for cotton and gram. For crops where the model is worse than
useless, alpha goes to zero and the rules take over.

Learning where your model fails, and routing around it, is a more defensible
contribution than a marginally better global score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def rank_norm(s: pd.Series) -> pd.Series:
    """Map any score to [0, 1] by rank, so two scales can be averaged."""
    if s.nunique() <= 1:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return s.rank(pct=True)


def alpha_from_skill(per_crop_rho: pd.DataFrame,
                     floor: float = config.ALPHA_RHO_FLOOR) -> dict[str, float]:
    """Per-crop blend weight from cross-validated within-crop Spearman.

    rho <= floor  -> alpha 0, the rules carry the recommendation entirely
    rho >= 0.6    -> alpha 1, the model is trusted
    in between    -> linear
    """
    alphas = {}
    for row in per_crop_rho.itertuples(index=False):
        rho = row.rho
        if not np.isfinite(rho) or rho <= floor:
            alphas[row.Crop] = 0.0
        else:
            alphas[row.Crop] = float(min(1.0, (rho - floor) / (0.6 - floor)))
    return alphas


def blend(
    df: pd.DataFrame,
    learned_col: str,
    rule_col: str,
    alphas: dict[str, float],
    ood_mask: np.ndarray | None = None,
    group_cols=("District", "Season"),
) -> pd.Series:
    """Blend the two scores per query group.

    Rank normalisation happens *within* a query group, because a rank is only
    meaningful against the other candidates for the same district and season.
    Rows flagged out-of-distribution fall back to the rule score alone.
    """
    out = pd.Series(np.nan, index=df.index, name="score_blended")
    for _, g in df.groupby(list(group_cols)):
        learned = rank_norm(g[learned_col])
        rules = rank_norm(g[rule_col])
        a = g["Crop"].map(alphas).fillna(0.0).to_numpy(dtype=float)
        out.loc[g.index] = a * learned.to_numpy() + (1 - a) * rules.to_numpy()

    if ood_mask is not None:
        rules_all = pd.Series(np.nan, index=df.index)
        for _, g in df.groupby(list(group_cols)):
            rules_all.loc[g.index] = rank_norm(g[rule_col]).to_numpy()
        out = out.where(~pd.Series(ood_mask, index=df.index), rules_all)
    return out


def alpha_report(alphas: dict[str, float], per_crop_rho: pd.DataFrame) -> pd.DataFrame:
    """Table of per-crop skill, weight, and which component actually decides."""
    rows = []
    for row in per_crop_rho.itertuples(index=False):
        a = alphas.get(row.Crop, 0.0)
        rows.append({
            "Crop": row.Crop, "n": row.n, "rho": row.rho, "alpha": round(a, 2),
            "decided_by": "rules" if a == 0 else ("model" if a >= 0.8 else "blend"),
        })
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)


def per_crop_ranking_skill(df: pd.DataFrame, score_col: str,
                           truth_col: str = "relevance") -> pd.DataFrame:
    """Per-crop skill of the *ranker*, measured the way the ranker is used.

    The alpha weights govern S1, which is a ranker, so they must come from
    ranking skill — not from the yield model's within-crop Spearman. Those are
    different models answering different questions, and using one to weight the
    other was a defect in the first implementation.

    For each crop, this asks: across the 34 districts, does a higher learned
    score go with a higher revealed relevance? That is exactly the judgement
    the blend is arbitrating.
    """
    from scipy.stats import spearmanr

    rows = []
    for crop, g in df.groupby("Crop"):
        if len(g) < 4 or g[score_col].nunique() < 2 or g[truth_col].nunique() < 2:
            rows.append({"Crop": crop, "n": len(g), "rho": np.nan})
            continue
        rho = spearmanr(g[truth_col], g[score_col]).statistic
        rows.append({"Crop": crop, "n": len(g),
                     "rho": round(float(rho), 3) if np.isfinite(rho) else np.nan})
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)
