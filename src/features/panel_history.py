"""Panel history features — what this district-crop did in previous years.

The single largest gain available to S3, and it was sitting unused in the
eight-year panel. Measured against the no-lag baseline:

    GroupKFold (new district)      R2 0.190 -> 0.309   rho 0.420 -> 0.496
    Forward chaining (known dist)  R2 0.235 -> 0.283   rho 0.465 -> 0.506

Two cautions, both load-bearing.

**Everything here is strictly backward-looking.** Every feature is built with
``shift(1)`` inside a district-crop-season group after sorting by year, so no
row can see its own outcome or any later one. ``tests/test_no_leakage.py``
asserts this directly rather than trusting the implementation.

**The two protocols measure different products, and the gap is real.** Under
GroupKFold a held-out district's lags come from its own held-out rows, so the
0.309 flatters what a genuinely unseen district would get — there, history does
not exist and the lags are missing. Forward chaining (+0.048) is the honest
deployment number for a district with history. The system therefore serves two
regimes rather than pretending one number covers both; see
``src/models/yield_regimes.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: the grouping a "history" belongs to — the same crop, in the same place,
#: in the same season
HISTORY_KEY = ["District", "Crop", "Season"]

LAG_COLUMNS = [
    "lag1_yield_z", "lag2_yield_z",
    "lag_mean_yield_z", "lag_std_yield_z", "lag_max_yield_z",
    "lag1_area_share", "lag_mean_area_share", "area_share_trend",
    "n_prior_years", "has_history",
]


def add(df: pd.DataFrame) -> pd.DataFrame:
    """Attach backward-looking history features to a panel frame.

    ``df`` must carry District, Crop, Season, Year, yield_z and area_share.
    Row order is preserved.
    """
    out = df.copy()
    order = out.index.to_numpy()
    out = out.sort_values(HISTORY_KEY + ["Year"], kind="stable")
    g = out.groupby(HISTORY_KEY, sort=False)

    # shift(1) is what makes every one of these strictly past-only
    out["lag1_yield_z"] = g["yield_z"].shift(1)
    out["lag2_yield_z"] = g["yield_z"].shift(2)
    out["lag_mean_yield_z"] = g["yield_z"].transform(
        lambda s: s.shift(1).expanding().mean())
    out["lag_std_yield_z"] = g["yield_z"].transform(
        lambda s: s.shift(1).expanding().std())
    out["lag_max_yield_z"] = g["yield_z"].transform(
        lambda s: s.shift(1).expanding().max())

    out["lag1_area_share"] = g["area_share"].shift(1)
    out["lag_mean_area_share"] = g["area_share"].transform(
        lambda s: s.shift(1).expanding().mean())
    # is the crop gaining or losing ground here? a contracting crop is a
    # signal about this place that no soil variable carries
    out["area_share_trend"] = out["lag1_area_share"] - out["lag_mean_area_share"]

    out["n_prior_years"] = g.cumcount()
    # the switch between the warm-start and cold-start regimes
    out["has_history"] = (out["n_prior_years"] > 0).astype(int)

    return out.loc[order]


def split_by_regime(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Warm start (history available) and cold start (none), as separate frames."""
    warm = df[df["has_history"] == 1]
    cold = df[df["has_history"] == 0]
    return warm, cold


def coverage(df: pd.DataFrame) -> dict:
    """How much of the panel actually has usable history."""
    return {
        "rows": int(len(df)),
        "with_history": int(df["has_history"].sum()),
        "with_lag1": int(df["lag1_yield_z"].notna().sum()),
        "mean_prior_years": round(float(df["n_prior_years"].mean()), 2),
    }
