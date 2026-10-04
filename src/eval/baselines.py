"""Reference baselines (plan §7.3) — "do not skip".

    "Very few crop-recommendation papers compute a popularity baseline at all;
     computing one and beating it is a stronger claim than a high accuracy
     number with no reference point."

The popularity prior scores **NDCG@5 = 0.644** and a yield-regression framing
scores *below* it at 0.216. Any ranking number reported without it beside it
is uninterpretable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.eval.splits import group_kfold


def random_baseline(df: pd.DataFrame, seed: int = config.SEED) -> pd.Series:
    """Uniform random scores — the floor, ~0.120 NDCG@5."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(len(df)), index=df.index, name="score_random")


def popularity_prior_oof(df: pd.DataFrame, n_splits: int = config.N_FOLDS) -> pd.Series:
    """Out-of-fold state-wide popularity — the bar to beat.

    For each test fold, a crop's score is the total area it occupies across the
    *training* districts only, within the same season. Computing it in-fold
    would leak the test districts' own planting decisions into their score, and
    would flatter the baseline the model has to clear.
    """
    scores = pd.Series(np.nan, index=df.index, name="score_popularity")

    for train_idx, test_idx in group_kfold(df, n_splits):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        prior = train.groupby(["Season", "Crop"])["Area"].sum()
        season_prior = train.groupby("Season")["Area"].sum()

        keys = list(zip(test["Season"], test["Crop"]))
        vals = [prior.get(k, 0.0) / season_prior.get(k[0], 1.0) for k in keys]
        scores.iloc[test_idx] = vals

    # Districts unseen in any training fold cannot happen under GroupKFold, but
    # a crop absent from every training district can — score it zero, not NaN.
    return scores.fillna(0.0)


def popularity_prior_lodo(df: pd.DataFrame) -> pd.Series:
    """The same prior under leave-one-district-out — the headline protocol."""
    scores = pd.Series(np.nan, index=df.index, name="score_popularity_lodo")
    for district in df["District"].unique():
        test = df["District"] == district
        train = df[~test]
        prior = train.groupby(["Season", "Crop"])["Area"].sum()
        season_prior = train.groupby("Season")["Area"].sum()
        keys = list(zip(df.loc[test, "Season"], df.loc[test, "Crop"]))
        scores.loc[test] = [prior.get(k, 0.0) / season_prior.get(k[0], 1.0) for k in keys]
    return scores.fillna(0.0)


def popularity_prior_split(df: pd.DataFrame, splitter) -> pd.Series:
    """The popularity prior under any splitter, computed from each fold's train rows.

    The same state-wide share of season area as ``popularity_prior_oof``, but
    for forward chaining and grouped + temporal as well: under forward chaining
    the prior for a test year comes only from earlier years, so it can never
    see the season it is scored on.
    """
    scores = pd.Series(np.nan, index=df.index, name="score_popularity")
    for train_idx, test_idx in splitter(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        prior = train.groupby(["Season", "Crop"])["Area"].sum()
        season_prior = train.groupby("Season")["Area"].sum()
        keys = list(zip(test["Season"], test["Crop"]))
        scores.iloc[test_idx] = [prior.get(k, 0.0) / season_prior.get(k[0], 1.0)
                                 for k in keys]
    return scores


def district_persistence(df: pd.DataFrame) -> pd.Series:
    """Last year's area share of the same crop, in the same district and season.

    The strongest trivial baseline a *warm-start* ranker faces: "grow what this
    district grew last year". Strictly backward-looking — a row sees only the
    previous year's share — so it is legitimate under forward chaining, and it
    is reported beside the popularity prior so a ranker cannot look good merely
    by beating the weaker of the two. A crop with no previous year scores 0.
    """
    order = df.sort_values("Year").index
    prev = (df.loc[order]
              .groupby(["District", "Season", "Crop"])["area_share"]
              .shift(1))
    return prev.reindex(df.index).fillna(0.0).rename("score_persistence")


def rule_score_baseline(df: pd.DataFrame) -> pd.Series:
    """The S2 agronomic scorer used as a standalone ranker.

    Needs no labels at all, which makes it the honest reference for "how far
    does pure agronomy get you". It is also the fallback S5 routes to when the
    learned model is out of its depth.
    """
    from src.data.training_set import district_features
    from src.ontology.crop_map import to_fertiliser_crop
    from src.rules.suitability import score

    feats = district_features().set_index("District")
    out = []
    for row in df.itertuples(index=False):
        fert_crop = to_fertiliser_crop(row.Crop)
        if fert_crop is None or row.District not in feats.index:
            out.append(0.0)
            continue
        f = feats.loc[row.District].to_dict()
        s = score(fert_crop, row.Season, f, irrigated=True)
        out.append(0.0 if not s.scored else float(s.score))
    return pd.Series(out, index=df.index, name="score_rules")
