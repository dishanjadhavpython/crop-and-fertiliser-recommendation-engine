"""Block D — spatial and cross-block interactions (plan §4.4).

Where domain knowledge enters the model. Worth +0.014 R2 on top of everything
else in the §7.1 ablation — small, but the largest gain available after the
soil blocks.

The kNN spatial features are built but default to *off*: the ablation shows
them costing 0.013 R2 at n=34. Same discipline as the contextual z-scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src import config

#: features smoothed over the 5 nearest talukas when spatial features are on
KNN_SOURCE_COLS = ["NI_N", "NI_P", "NI_K", "NI_OC", "rain_annual",
                   "rootzone_awc", "aridity_annual", "def_Zn"]


def build(df: pd.DataFrame, with_spatial: bool = False) -> pd.DataFrame:
    """Add Block D columns to a frame already carrying Blocks A-C.

    ``df`` must contain the Block A, B and C columns and the key columns.
    """
    out = df.copy()

    # Rain the soil can actually retain — sandy soil wastes heavy rain.
    out["water_supply"] = out["rain_Kharif"] * out["rootzone_awc"] / 1000.0

    # Deep clay buffers a 15-day break; shallow murum does not.
    out["drought_vuln"] = out["dry_spell_Kharif"] / (out["rootzone_awc"] / 100.0)

    # Nitrogen loss risk. Feeds the S4 split-dose schedule directly.
    out["leach_risk"] = (
        out["rain_annual"] * (1 - out["NI_OC"] / 3.0) * (200.0 - out["awc_mm_m"])
    ) / 1e4

    # Phosphorus locks up above pH 8 and below 5.5, so the raw P index
    # overstates what the plant can actually reach.
    out["p_availability"] = out["NI_P"] * _ph_penalty(out)

    # Zinc deficiency is pH-driven; alkaline AND Zn-deficient compounds.
    out["zn_lockout"] = out["def_Zn"] * out["ph_alkaline"] / 100.0

    # Salinity is manageable with good drainage and severe without it.
    out["salinity_x_drain"] = out["ec_saline"] / out["drainage_ord"]

    # The Rabi water deficit a farmer must supply.
    out["irrig_need"] = np.clip(-out["cwb_Rabi"], 0, None)
    out["irrig_need_summer"] = np.clip(-out["cwb_Summer"], 0, None)

    # A few more that pair a soil constraint with a climate driver.
    out["n_leach_pressure"] = out["leach_risk"] * (3.0 - out["NI_N"])
    out["heat_x_shallow"] = out["heat_days_Summer"] * out["is_shallow"]
    out["fungal_x_humid_soil"] = out["fungal_days_Kharif"] * (out["drainage_ord"] <= 3).astype(int)
    out["waterlog_risk"] = out["rain_Kharif"] / (out["drainage_ord"] * out["depth_mm"] / 1000.0)
    out["moisture_stress_rabi"] = out["irrig_need"] / (out["rootzone_awc"] + 1.0)
    out["fertility_x_water"] = out["macro_NI"] * out["water_supply"]
    out["micro_burden"] = out["micro_def_count"] * out["micro_def_worst_pct"] / 100.0
    out["organic_deficit"] = (3.0 - out["NI_OC"]) * out["ph_stress"] / 100.0

    if with_spatial:
        out = add_spatial(out)
    return out


def _ph_penalty(df: pd.DataFrame) -> pd.Series:
    """Fraction of P that stays plant-available given the pH distribution.

    Alkaline soils fix P as calcium phosphate, acidic soils as iron/aluminium
    phosphate. Neutral samples pass through unpenalised.
    """
    return (df["ph_neutral"] + 0.55 * df["ph_alkaline"] + 0.7 * df["ph_acidic"]) / 100.0


def add_spatial(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """5-nearest-neighbour means and local anomalies (plan §4.4, flagged off).

    Encodes regional agro-ecology and how a taluka departs from it. Neighbours
    are found on the great-circle distance between taluka centroids.
    """
    out = df.copy()
    coords = np.radians(out[["Latitude", "Longitude"]].to_numpy(dtype=float))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(coords)
    _, idx = nn.kneighbors(coords)
    neigh = idx[:, 1:]      # drop self

    for col in KNN_SOURCE_COLS:
        if col not in out.columns:
            continue
        vals = out[col].to_numpy(dtype=float)
        knn_mean = vals[neigh].mean(axis=1)
        out[f"knn5_{col}"] = knn_mean
        out[f"anom_{col}"] = vals - knn_mean
    return out
