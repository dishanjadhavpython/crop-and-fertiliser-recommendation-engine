"""Block B — soil physical properties (plan §4.2).

Six categorical columns. With 34 effective samples, one-hot encoding is
wasteful and target encoding overfits, so each is converted to a physically
meaningful number instead. This was worth +0.065 R2 in the §7.1 ablation — the
second-largest gain of any block.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.load import taluka_universe

#: coarse-fragment penalty already folded into the skeletal AWC values
_TEXTURE_ORD = {"Sandy": 1, "Sandy loam": 2, "Loamy-skeletal": 3,
                "Loamy": 4, "Clayey-skeletal": 5, "Clayey": 6}


@lru_cache(maxsize=8)
def build() -> pd.DataFrame:
    """Return the Block B feature frame, one row per taluka."""
    u = taluka_universe()
    out = u[config.KEY].copy()

    # Four Alluvial talukas (Tumsar, Ausa, Latur, Parseoni) carry no recorded
    # texture. Imputed from the modal texture of their own soil type and
    # flagged, so a downstream consumer can see the value was inferred.
    texture = u["Soil_Texture"]
    out["texture_imputed"] = texture.isna().astype(int)
    modal = (u.dropna(subset=["Soil_Texture"])
               .groupby("Soil_Type")["Soil_Texture"]
               .agg(lambda s: s.mode().iat[0]))
    texture = texture.fillna(u["Soil_Type"].map(modal))

    out["awc_mm_m"] = texture.map(config.AWC_MM_PER_M)
    out["depth_mm"] = u["Soil_Depth"].map(config.DEPTH_MM)
    out["drainage_ord"] = u["Soil_Drainage"].map(config.DRAINAGE_ORD)
    out["texture_ord"] = texture.map(_TEXTURE_ORD)
    out["ph_class_value"] = u["Soil_pH_Class"].map(config.PH_CLASS_VALUE)

    # The most agronomically loaded feature in the store: total mm of water the
    # root zone can actually hold.
    out["rootzone_awc"] = out["awc_mm_m"] * out["depth_mm"] / 1000.0

    # How homogeneous the taluka is. Low purity means lower confidence in every
    # other soil feature, so it doubles as a trust weight.
    out["soil_purity"] = u["Soil_Type_Share_pct"]
    out["soil_points_sampled"] = u["Soil_Points_Sampled"]

    # A handful of low-cardinality classes carry real agronomy and are cheap as
    # binary indicators — far cheaper than one-hot over every level.
    out["is_black_soil"] = (u["Soil_Type"] == "Black (Regur)").astype(int)
    out["is_saline_alkaline"] = (
        (u["Soil_Type"] == "Saline / Alkaline")
        | (u["Soil_Type_Secondary"] == "Saline / Alkaline")
    ).astype(int)
    out["is_lateritic"] = (u["Soil_Type"].isin(["Laterite", "Red & Yellow"])).astype(int)
    out["is_vertisol"] = (u["Soil_Order"] == "Vertisols").astype(int)
    out["is_skeletal"] = texture.str.contains("skeletal", na=False).astype(int)
    out["is_shallow"] = u["Soil_Depth"].isin(["Very shallow", "Shallow"]).astype(int)

    out["Latitude"] = u["Latitude"]
    out["Longitude"] = u["Longitude"]

    # "Others" in Soil_pH_Class is a genuine unknown; fall back to the state
    # median rather than inventing a value or dropping the taluka.
    out["ph_class_value"] = out["ph_class_value"].fillna(out["ph_class_value"].median())
    return out
