"""Crop-conditional agronomic fit features (early fusion of S2 into S1).

The ranker's features are all *district* properties: rainfall, depth, pH. The
crop enters only as ``crop_id``, so a tree has to rediscover "rice wants
water, pearl millet does not" from 34 districts of evidence it does not have.

The S2 suitability scorer already knows this, from FAO EcoCrop envelopes and
**no labels whatsoever**. Feeding its per-factor scores in as features is
therefore free of leakage by construction, and strictly more expressive than
blending the two scores after the fact: the ranker can learn *when* the rules
are worth trusting rather than being told once per crop.

This is the difference between late fusion (a fixed per-crop alpha) and early
fusion (the model decides), and it is the largest unexploited gain in the
system.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.ontology.crop_map import to_fertiliser_crop
from src.rules.crop_requirements import get as get_envelope
from src.rules.suitability import FACTORS, effective_water, score

#: one column per Liebig factor, plus the aggregate and its argmin
FIT_COLUMNS = ([f"fit_{f}" for f in FACTORS]
               + ["fit_min", "fit_mean", "fit_gap", "fit_vetoed",
                  "fit_has_envelope", "fit_water_ratio", "fit_temp_ratio",
                  "fit_lgp_ratio", "fit_depth_ratio", "fit_duration_days"])


def _ratio(value: float, lo: float, hi: float) -> float:
    """Where the district sits inside the crop's optimal band.

    0 at the lower optimum, 1 at the upper, negative below and >1 above. This
    is signed distance-to-optimum, which a tree can split on far more cheaply
    than it can reconstruct the same thing from raw rainfall plus crop_id.
    """
    if not np.isfinite(value) or hi <= lo:
        return np.nan
    return (value - lo) / (hi - lo)


def compute(df: pd.DataFrame, feature_source: pd.DataFrame,
            key: str = "District", irrigated: bool = True) -> pd.DataFrame:
    """Per-row agronomic fit features for a (key, Season, Crop) frame.

    ``feature_source`` is indexed by ``key`` and carries the agro-climatic and
    soil columns the suitability scorer reads.
    """
    src = feature_source.set_index(key) if key in feature_source.columns else feature_source
    cache: dict[tuple, dict] = {}
    rows = []

    for k, season, crop in zip(df[key], df["Season"], df["Crop"]):
        ck = (k, season, crop)
        if ck not in cache:
            cache[ck] = _one(k, season, crop, src, irrigated)
        rows.append(cache[ck])

    return pd.DataFrame(rows, index=df.index)


def _one(key, season: str, crop: str, src: pd.DataFrame, irrigated: bool) -> dict:
    blank = {c: np.nan for c in FIT_COLUMNS}
    blank["fit_has_envelope"] = 0.0
    blank["fit_vetoed"] = 0.0

    fert_crop = to_fertiliser_crop(crop)
    if fert_crop is None or key not in src.index:
        return blank
    req = get_envelope(fert_crop)
    if req is None:
        return blank

    feats = src.loc[key].to_dict()
    s = score(fert_crop, season, feats, irrigated=irrigated)

    out = dict(blank)
    out["fit_has_envelope"] = 1.0
    out["fit_vetoed"] = float(bool(s.vetoed))

    if not s.factors:
        # out-of-season: every factor is inapplicable, and that is the signal
        out["fit_min"] = 0.0
        out["fit_mean"] = 0.0
        out["fit_gap"] = 0.0
        out["fit_duration_days"] = float(req.duration_days)
        return out

    for f in FACTORS:
        out[f"fit_{f}"] = float(s.factors[f])
    vals = np.array([s.factors[f] for f in FACTORS], dtype=float)
    out["fit_min"] = float(vals.min())
    out["fit_mean"] = float(vals.mean())
    # how far the binding constraint sits below the rest — a crop held back by
    # one factor is a different proposition from one that is uniformly mediocre
    out["fit_gap"] = float(vals.mean() - vals.min())

    water = effective_water(feats, season)
    tmax = feats.get(f"tmax_mean_{season}", feats.get("tmax_mean_annual", np.nan))
    tmin = feats.get(f"tmin_mean_{season}", feats.get("tmin_mean_annual", np.nan))
    lgp = feats.get(f"lgp_{season}", feats.get("lgp_annual", np.nan))

    out["fit_water_ratio"] = _ratio(water, req.rain_mm.opt_min, req.rain_mm.opt_max)
    out["fit_temp_ratio"] = _ratio((tmax + tmin) / 2.0, req.temp_c.opt_min, req.temp_c.opt_max)
    out["fit_lgp_ratio"] = float(lgp / req.min_lgp_days) if req.min_lgp_days else np.nan
    out["fit_depth_ratio"] = float(feats.get("depth_mm", np.nan) / req.min_depth_mm)
    out["fit_duration_days"] = float(req.duration_days)
    return out
