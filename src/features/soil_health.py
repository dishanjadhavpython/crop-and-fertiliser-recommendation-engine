"""Block A — the 12 Soil Health Card components (plan §4.1).

The SHC gives each component as a *distribution over sample percentages*, not a
value, which needs specific handling:

* **Nutrient Index** collapses a Low/Medium/High triplet to the bounded 1-3
  scalar ICAR and the SHC scheme themselves use.
* **Isometric log-ratio** handles the fact that the triplet sums to 100 by
  construction. Feeding all three parts to a model creates exact collinearity
  and invites it to learn from an artefact of closure; ILR maps the 3-part
  simplex to two unconstrained real coordinates.
* **Deficiency share only** for the six micronutrients: they are a 2-part
  composition, so one number carries all the information and keeping
  ``Sufficient_pct`` would add a perfectly redundant column to a model already
  short on samples.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_shc

#: components published as Low / Medium / High percentages
LMH_COMPONENTS = {
    "N": "Nitrogen_N",
    "P": "Phosphorus_P",
    "K": "Potassium_K",
    "OC": "OrganicCarbon_OC",
}
#: components published as Sufficient / Deficient percentages
SUF_DEF_COMPONENTS = {
    "S": "Sulphur_S",
    "Fe": "Iron_Fe",
    "Zn": "Zinc_Zn",
    "Cu": "Copper_Cu",
    "B": "Boron_B",
    "Mn": "Manganese_Mn",
}
#: the seven core soil variables used for contextual z-scores
CORE_SOIL_VARS = ["NI_N", "NI_P", "NI_K", "NI_OC", "ph_stress", "ec_saline", "def_S"]


def nutrient_index(low: pd.Series, med: pd.Series, high: pd.Series) -> pd.Series:
    """NI = (1*Low% + 2*Med% + 3*High%) / 100, bounded [1, 3]."""
    return (1 * low + 2 * med + 3 * high) / 100.0


def ilr_3part(low, med, high) -> tuple[pd.Series, pd.Series]:
    """Isometric log-ratio of a 3-part composition (L, M, H).

    A ``(x + 0.5) / 101.5`` closure is applied first: several talukas record a
    true 0% in a band, and log(0) is undefined.
    """
    l = (np.asarray(low, dtype=float) + 0.5) / 101.5
    m = (np.asarray(med, dtype=float) + 0.5) / 101.5
    h = (np.asarray(high, dtype=float) + 0.5) / 101.5
    ilr1 = np.sqrt(2.0 / 3.0) * np.log(h / np.sqrt(l * m))
    ilr2 = np.sqrt(1.0 / 2.0) * np.log(m / l)
    return pd.Series(ilr1), pd.Series(ilr2)


@lru_cache(maxsize=8)
def build(with_context_z: bool = False, cycle: str | None = None) -> pd.DataFrame:
    """Return the Block A feature frame, one row per taluka.

    ``with_context_z`` adds the state- and district-relative z-scores. They are
    off by default: the §7.1 ablation shows them costing 0.005 R2 at n=34.
    Build them, keep them behind the flag, switch them on with multi-year labels.
    """
    shc = load_shc(cycle)
    out = shc[config.KEY].copy()
    out["Cycle"] = shc["Cycle"].to_numpy()

    for short, prefix in LMH_COMPONENTS.items():
        low = shc[f"{prefix}_Low_pct"]
        med = shc[f"{prefix}_Medium_pct"]
        high = shc[f"{prefix}_High_pct"]
        out[f"NI_{short}"] = nutrient_index(low, med, high).to_numpy()
        i1, i2 = ilr_3part(low, med, high)
        out[f"ilr1_{short}"] = i1.to_numpy()
        out[f"ilr2_{short}"] = i2.to_numpy()
        out[f"low_{short}"] = low.to_numpy()
        out[f"high_{short}"] = high.to_numpy()
        out[f"nsamp_{short}"] = shc[f"{prefix}_Samples"].to_numpy()

    # pH is a 3-part composition too (acidic / neutral / alkaline)
    out["ph_acidic"] = shc["pH_Acidic_pct"].to_numpy()
    out["ph_neutral"] = shc["pH_Neutral_pct"].to_numpy()
    out["ph_alkaline"] = shc["pH_Alkaline_pct"].to_numpy()
    # deviation from ideal in either direction, as one variable
    out["ph_stress"] = 100.0 - shc["pH_Neutral_pct"].to_numpy()
    i1, i2 = ilr_3part(shc["pH_Acidic_pct"], shc["pH_Neutral_pct"], shc["pH_Alkaline_pct"])
    out["ilr1_pH"], out["ilr2_pH"] = i1.to_numpy(), i2.to_numpy()

    out["ec_saline"] = shc["EC_Saline_pct"].to_numpy()

    # six micronutrients: keep the deficient share only (2-part composition)
    for short, prefix in SUF_DEF_COMPONENTS.items():
        out[f"def_{short}"] = shc[f"{prefix}_Deficient_pct"].to_numpy()

    # ---- composites and ratios (plan §4.1) ---------------------------------
    ni = out[["NI_N", "NI_P", "NI_K"]]
    out["macro_NI"] = ni.mean(axis=1)
    # a soil high in N but low in P behaves differently from a uniformly medium
    # one; the mean hides that, the spread exposes it
    out["npk_imbalance"] = ni.std(axis=1, ddof=0)
    out["ratio_NP"] = out["NI_N"] / out["NI_P"]
    out["ratio_NK"] = out["NI_N"] / out["NI_K"]
    out["ratio_PK"] = out["NI_P"] / out["NI_K"]

    def_cols = [f"def_{m}" for m in config.MICRONUTRIENTS]
    defs = out[def_cols]
    out["micro_def_mean"] = defs.mean(axis=1)
    # multi-deficiency soils need a different intervention than single-deficiency ones
    out["micro_def_count"] = (defs > config.MICRO_DEFICIENCY_TRIGGER_PCT).sum(axis=1)
    # names the binding constraint — drives the S4 correction layer directly
    out["micro_def_worst"] = defs.idxmax(axis=1).str.replace("def_", "", regex=False)
    out["micro_def_worst_pct"] = defs.max(axis=1)

    # sample count is a feature AND a training weight: a taluka with 4,480
    # samples deserves more trust than one with 200
    samples = shc[[f"{p}_Samples" for p in LMH_COMPONENTS.values()]].mean(axis=1)
    out["n_samples"] = samples.to_numpy()
    out["n_samples_log"] = np.log1p(samples).to_numpy()

    if with_context_z:
        out = add_contextual_z(out)
    return out


def add_contextual_z(df: pd.DataFrame) -> pd.DataFrame:
    """State- and district-relative z-scores (plan §4.1, kept behind a flag).

    The district version is the interesting one: it asks "is this taluka better
    or worse than its neighbours", which is how an extension officer reasons.
    """
    out = df.copy()
    for col in CORE_SOIL_VARS:
        if col not in out.columns:
            continue
        s = out[col]
        out[f"z_state_{col}"] = (s - s.mean()) / (s.std(ddof=0) or 1.0)
        grp = out.groupby("District")[col]
        out[f"z_district_{col}"] = ((s - grp.transform("mean")) /
                                    grp.transform("std").replace(0, np.nan).fillna(1.0))
    return out
