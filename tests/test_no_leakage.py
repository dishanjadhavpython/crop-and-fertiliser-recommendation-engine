"""The other day-one test (plan §10).

    "Leakage test: assert that no district appears in both the train and test
     index of any fold. Run it in CI. Most leaks are introduced later by
     someone refactoring a split."

Also guards the two subtler leaks this design is exposed to: grading relevance
globally instead of within a district, and selecting features on the full
dataset before cross-validating.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data.training_set import build, model_features
from src.eval import baselines, splits
from src.models import ranker, yield_quantile

DF = build()
Y = yield_quantile.training_rows(DF)


@pytest.mark.parametrize("splitter", [
    splits.group_kfold,
    splits.leave_one_district_out,
    splits.spatial_block_folds,
])
def test_no_district_appears_in_both_train_and_test(splitter):
    districts = DF["District"].to_numpy()
    n_folds = 0
    for train_idx, test_idx in splitter(DF):
        n_folds += 1
        overlap = set(districts[train_idx]) & set(districts[test_idx])
        assert not overlap, f"{splitter.__name__} leaked {overlap}"
        assert len(train_idx) and len(test_idx)
    assert n_folds >= 2


def test_every_row_is_tested_exactly_once():
    for splitter in (splits.group_kfold, splits.leave_one_district_out):
        seen = np.concatenate([te for _, te in splitter(DF)])
        assert len(seen) == len(DF)
        assert len(set(seen)) == len(DF)


def test_leave_one_district_out_has_one_fold_per_district():
    folds = list(splits.leave_one_district_out(DF))
    assert len(folds) == DF["District"].nunique() == 34
    for _, test_idx in folds:
        assert DF["District"].iloc[test_idx].nunique() == 1


def test_ranker_groups_are_contiguous_after_sorting():
    """LightGBM assigns documents to queries by position; unsorted rows corrupt it."""
    idx = np.arange(len(DF))
    sorted_idx, sizes = ranker._grouped(DF, idx)
    keys = splits.ranking_groups(DF)[sorted_idx]
    assert sizes.sum() == len(DF)
    # each group's slice must contain exactly one distinct query key
    start = 0
    for size in sizes:
        assert len(set(keys[start:start + size])) == 1
        start += size


def test_relevance_is_graded_within_district_not_globally():
    """Global grading would leak a district's overall scale into its labels."""
    planted = DF[DF["planted"] == 1]
    per_query = planted.groupby(["District", "Season"])["relevance"].nunique()
    # a global grading would give small districts a single grade; within-group
    # grading spreads them across the scale
    assert per_query.mean() > 2.5
    assert planted["relevance"].min() >= 1
    assert DF.loc[DF["planted"] == 0, "relevance"].eq(0).all()


def test_popularity_prior_is_computed_out_of_fold():
    """An in-fold prior would leak the test districts' own planting decisions."""
    src = inspect.getsource(baselines.popularity_prior_oof)
    assert "group_kfold" in src
    scores = baselines.popularity_prior_oof(DF)
    assert scores.notna().all()
    # a crop's prior must differ across folds, or it was computed globally
    per_crop = DF.assign(p=scores).groupby(["Season", "Crop"])["p"].nunique()
    assert per_crop.max() > 1


def test_feature_selection_happens_inside_the_fold():
    """Selecting on the whole dataset then cross-validating is a leak."""
    src = inspect.getsource(yield_quantile.cross_val_quantiles)
    select_call = src.index("select_features")
    loop_start = src.index("for train_idx, test_idx in group_kfold")
    assert select_call > loop_start, "select_features must be called inside the CV loop"

    sig = inspect.signature(yield_quantile.select_features)
    assert "train" in sig.parameters


def test_selected_features_differ_across_folds():
    """If selection were global, every fold would choose the same features."""
    feats = model_features(Y)
    chosen = [
        tuple(yield_quantile.select_features(Y.iloc[tr], feats, 20))
        for tr, _ in splits.group_kfold(Y)
    ]
    assert len(set(chosen)) > 1


def test_target_columns_are_never_used_as_features():
    feats = set(model_features(DF))
    forbidden = {"Area", "Production", "Yield", "yield_z", "relevance",
                 "area_share", "area_share_district", "crop_yield_mean",
                 "crop_yield_std", "planted", "has_yield"}
    assert not (feats & forbidden)


def test_yield_target_is_within_crop_z_scored():
    """Training on raw yield is the documented path to the fake R2 = 0.93."""
    by_crop = Y.groupby("Crop")["yield_z"].mean().abs()
    assert (by_crop < 0.35).all(), "yield_z is not centred within crop"
    assert Y["Yield"].std() > 10        # raw yield spans sugarcane to sesamum
    assert 0.8 < Y["yield_z"].std() < 1.2
