"""Within-district heterogeneity, and taluka-level suitability distributions.

Aggregating 351 talukas to 34 districts by a weighted mean destroys real
information: a district that is uniformly deep black soil and one that is half
deep black and half shallow laterite produce the same district mean, yet they
support completely different cropping patterns.

Two blocks recover it, and neither touches a label:

``disp_*``
    Dispersion of the key soil and climate variables across a district's own
    talukas — spread, range and skew, not just the centre.

``suitfrac_*``
    The share of a district's talukas where a given crop scores S1/S2/S3/N
    under the S2 gate. This is the strongest of the two: it is evaluated at
    the 351-taluka grain the features actually live at, and answers "how much
    of this district can grow this crop" rather than "would the average hectare
    of this district grow it".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.features.build_store import build_feature_store
from src.ontology.crop_map import to_fertiliser_crop
from src.rules.crop_requirements import get as get_envelope
from src.rules.suitability import score

#: variables whose within-district spread carries agronomic meaning
DISPERSION_VARS = [
    "NI_N", "NI_P", "NI_K", "NI_OC", "ph_stress", "ec_saline",
    "rootzone_awc", "depth_mm", "awc_mm_m", "drainage_ord",
    "rain_annual", "rain_Kharif", "aridity_annual", "lgp_annual",
    "dry_spell_Kharif", "leach_risk", "def_Zn", "def_S",
]


def dispersion_features(store: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per district: spread of each variable across its own talukas."""
    store = build_feature_store() if store is None else store
    rows = []
    for district, g in store.groupby("District", sort=True):
        rec = {"District": district}
        for v in DISPERSION_VARS:
            if v not in g.columns:
                continue
            x = g[v].to_numpy(dtype=float)
            rec[f"disp_sd_{v}"] = float(np.std(x, ddof=0))
            rec[f"disp_range_{v}"] = float(x.max() - x.min())
            # coefficient of variation makes spreads comparable across units
            m = float(np.mean(x))
            rec[f"disp_cv_{v}"] = float(np.std(x, ddof=0) / m) if m else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def suitability_fractions(store: pd.DataFrame | None = None,
                          irrigated: bool = True) -> pd.DataFrame:
    """Share of each district's talukas in each suitability class, per crop-season.

    Label-free by construction: the S2 gate reads only soil and climate.
    """
    from src.data.load import load_apy

    store = build_feature_store() if store is None else store
    crops = sorted(load_apy()["Crop"].unique())
    seasons = config.APY_SEASONS

    rows = []
    for district, g in store.groupby("District", sort=True):
        taluka_feats = [r.to_dict() for _, r in g.iterrows()]
        n = len(taluka_feats)
        for crop in crops:
            fert = to_fertiliser_crop(crop)
            has_env = fert is not None and get_envelope(fert) is not None
            for season in seasons:
                rec = {"District": district, "Crop": crop, "Season": season,
                       "suitfrac_n_talukas": n}
                if not has_env:
                    rows.append({**rec, "suitfrac_S1": np.nan, "suitfrac_S2plus": np.nan,
                                 "suitfrac_viable": np.nan, "suitfrac_vetoed": np.nan,
                                 "suitfrac_mean_score": np.nan,
                                 "suitfrac_best_score": np.nan,
                                 "suitfrac_score_sd": np.nan})
                    continue
                scores, classes = [], []
                for f in taluka_feats:
                    s = score(fert, season, f, irrigated=irrigated)
                    scores.append(0.0 if not s.scored else float(s.score))
                    classes.append(s.suitability_class)
                arr = np.array(scores, dtype=float)
                rows.append({
                    **rec,
                    "suitfrac_S1": float(np.mean([c == "S1" for c in classes])),
                    "suitfrac_S2plus": float(np.mean([c in ("S1", "S2") for c in classes])),
                    "suitfrac_viable": float(np.mean([c != "N" for c in classes])),
                    "suitfrac_vetoed": float(np.mean([c == "N" for c in classes])),
                    "suitfrac_mean_score": float(arr.mean()),
                    "suitfrac_best_score": float(arr.max()),
                    # a district split between excellent and impossible land is
                    # a different proposition from a uniformly mediocre one
                    "suitfrac_score_sd": float(arr.std(ddof=0)),
                })
    return pd.DataFrame(rows)


SUITFRAC_COLUMNS = ["suitfrac_S1", "suitfrac_S2plus", "suitfrac_viable",
                    "suitfrac_vetoed", "suitfrac_mean_score",
                    "suitfrac_best_score", "suitfrac_score_sd", "suitfrac_n_talukas"]
DISPERSION_COLUMNS_PREFIX = "disp_"
