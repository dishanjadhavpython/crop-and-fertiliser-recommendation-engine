"""Cross-validation schemes (plan §7, rule one).

    "Talukas in the same district share soil, climate and — critically — the
     same district-level label. A random split puts sibling talukas in train
     and test, and the model scores well by recognising the district rather
     than the agronomy."

Every split here groups by district. Nothing in this project may use
``train_test_split`` or an ungrouped ``KFold``; the leakage test in
``tests/test_no_leakage.py`` enforces that on the splitters themselves.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from src import config


def district_groups(df: pd.DataFrame) -> np.ndarray:
    """The grouping variable for every split in the project."""
    return df["District"].to_numpy()


def group_kfold(df: pd.DataFrame, n_splits: int = config.N_FOLDS) -> Iterator[tuple]:
    """GroupKFold by district — the workhorse protocol."""
    groups = district_groups(df)
    yield from GroupKFold(n_splits=n_splits).split(df, groups=groups)


def leave_one_district_out(df: pd.DataFrame) -> Iterator[tuple]:
    """34 folds, one per district — the headline protocol.

    With n=34 this is affordable, and it is the most convincing scheme
    available for this data.
    """
    yield from LeaveOneGroupOut().split(df, groups=district_groups(df))


def spatial_block_folds(
    df: pd.DataFrame,
    n_blocks: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> Iterator[tuple]:
    """Geographic block CV — a stronger claim than GroupKFold alone.

    Districts are clustered on their centroid, so a held-out fold is a
    contiguous region rather than a scatter of districts. This breaks spatial
    autocorrelation that district grouping alone leaves intact.
    """
    centroids = (
        df.groupby("District")[["Latitude", "Longitude"]].mean().sort_index()
    )
    labels = KMeans(n_clusters=n_blocks, random_state=seed, n_init=10).fit_predict(centroids)
    block_of = dict(zip(centroids.index, labels))
    blocks = df["District"].map(block_of).to_numpy()

    idx = np.arange(len(df))
    for b in range(n_blocks):
        test = idx[blocks == b]
        train = idx[blocks != b]
        if len(test) and len(train):
            yield train, test


def ranking_groups(df: pd.DataFrame) -> np.ndarray:
    """Query keys for the learning-to-rank objective.

    Year joins the key on a panel: a farmer choosing in 2018 competes against
    2018's alternatives, not against a pooled decade.
    """
    from src.eval.metrics import query_columns

    cols = query_columns(df)
    key = df[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + "|" + df[c].astype(str)
    return key.to_numpy()


def group_sizes(df: pd.DataFrame, index: np.ndarray) -> np.ndarray:
    """LightGBM ranker group sizes for a fold, in the fold's own row order.

    The rows must already be sorted by query key or LightGBM will mis-assign
    documents to queries; ``sort_by_group`` below guarantees that.
    """
    keys = ranking_groups(df)[index]
    _, counts = np.unique(keys, return_counts=True)
    return counts


def sort_by_group(df: pd.DataFrame, index: np.ndarray) -> np.ndarray:
    """Reorder fold indices so rows of the same query are contiguous."""
    keys = ranking_groups(df)[index]
    return index[np.argsort(keys, kind="stable")]


# ---------------------------------------------------------------------------
# Temporal validation — only possible now that the panel spans eight years.
# ---------------------------------------------------------------------------
def forward_chaining(df: pd.DataFrame, min_train_years: int = 4) -> Iterator[tuple]:
    """Expanding-window split: train on the past, test on the next year.

    This is the protocol that answers the question a deployed recommender
    actually faces — "given everything up to last season, what should be sown
    next season?" — and it is the one the original single-year data could not
    support at all. A model that scores well under GroupKFold but badly here is
    memorising the era, not learning agronomy.
    """
    years = sorted(df["Year"].unique())
    idx = np.arange(len(df))
    for i in range(min_train_years, len(years)):
        train = idx[df["Year"].isin(years[:i]).to_numpy()]
        test = idx[(df["Year"] == years[i]).to_numpy()]
        if len(train) and len(test):
            yield train, test


def temporal_holdout(df: pd.DataFrame,
                     holdout_years: list[str] | None = None) -> Iterator[tuple]:
    """A single past/future cut — train on the early years, test on the last two."""
    holdout = holdout_years or config.TEMPORAL_HOLDOUT_YEARS
    idx = np.arange(len(df))
    mask = df["Year"].isin(holdout).to_numpy()
    train, test = idx[~mask], idx[mask]
    if len(train) and len(test):
        yield train, test


def grouped_and_temporal(df: pd.DataFrame,
                         holdout_years: list[str] | None = None) -> Iterator[tuple]:
    """The strictest protocol available: unseen districts AND unseen years.

    A fold's test set shares neither a district nor a year with its training
    set, so neither spatial autocorrelation nor an era effect can carry it.
    """
    holdout = holdout_years or config.TEMPORAL_HOLDOUT_YEARS
    idx = np.arange(len(df))
    future = df["Year"].isin(holdout).to_numpy()
    districts = district_groups(df)
    for train_idx, test_idx in GroupKFold(n_splits=config.N_FOLDS).split(df, groups=districts):
        train = np.intersect1d(train_idx, idx[~future])
        test = np.intersect1d(test_idx, idx[future])
        if len(train) and len(test):
            yield train, test
