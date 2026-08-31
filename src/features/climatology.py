"""Multi-year agro-climatology: normals, variability and SHC trend.

The original plan was sized for a single weather year and said so plainly:

    "With one weather year there is no inter-annual variation, so no model
     here can learn how crops respond to a dry year versus a wet one."

Three genuine weather years do not lift that restriction entirely — matching a
yield year to its own weather is still impossible, because the crop panel
(2015-16..2022-23) and the weather (2023-24..2025-26) do not overlap at all.
What three years *do* buy is an honest **normal** instead of a one-year
snapshot, and a first measure of **how variable** a place is, which is a
genuine agro-climatic property in its own right: two talukas with the same
mean rainfall but different variability are not equally risky to farm.

Two blocks:

``clim_*``   mean of each agro-climatic feature across the weather years
``var_*``    inter-annual spread — new information, impossible with one year

Plus ``shcTrend_*``: the direction each soil component is moving across the
three Soil Health Card cycles.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.features import agroclimate, soil_health

#: features whose year-to-year variability is agronomically meaningful
VARIABILITY_VARS = [
    "rain_annual", "rain_Kharif", "rain_Rabi",
    "dry_spell_Kharif", "monsoon_onset_doy", "lgp_annual",
    "aridity_annual", "cwb_Rabi", "gdd_annual",
    "heat_days_Summer", "severe_heat_days_Summer", "fungal_days_Kharif",
]

#: soil components whose cycle-to-cycle trend is worth carrying
TREND_VARS = ["NI_N", "NI_P", "NI_K", "NI_OC", "ph_stress", "ec_saline",
              "def_S", "def_Fe", "def_Zn", "def_B", "def_Mn", "macro_NI"]


@lru_cache(maxsize=1)
def weather_panel() -> pd.DataFrame:
    """Every weather year's Block C features, stacked."""
    return pd.concat([agroclimate.build(y) for y in config.WEATHER_YEARS],
                     ignore_index=True)


def climatology(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-taluka normals and inter-annual variability across the weather years."""
    panel = weather_panel() if panel is None else panel
    value_cols = [c for c in panel.columns
                  if c not in config.KEY + ["weather_year"]
                  and pd.api.types.is_numeric_dtype(panel[c])]

    g = panel.groupby(config.KEY, sort=True)
    out = g[value_cols].mean().reset_index()
    out.columns = config.KEY + [f"clim_{c}" for c in value_cols]

    # --- inter-annual variability: the genuinely new signal -----------------
    var_cols = [c for c in VARIABILITY_VARS if c in value_cols]
    sd = g[var_cols].std(ddof=0).reset_index()
    sd.columns = config.KEY + [f"var_sd_{c}" for c in var_cols]
    out = out.merge(sd, on=config.KEY, how="left")

    rng = (g[var_cols].max() - g[var_cols].min()).reset_index()
    rng.columns = config.KEY + [f"var_range_{c}" for c in var_cols]
    out = out.merge(rng, on=config.KEY, how="left")

    # relative variability: a 200 mm swing means something different in the
    # Konkan than it does in Marathwada
    for c in var_cols:
        mean = out[f"clim_{c}"].replace(0, np.nan)
        out[f"var_cv_{c}"] = (out[f"var_sd_{c}"] / mean.abs()).fillna(0.0)

    # how often the monsoon underperformed its own normal, in years
    if "rain_Kharif" in value_cols:
        piv = panel.pivot_table(index=config.KEY, columns="weather_year",
                                values="rain_Kharif")
        deficit = piv.lt(piv.mean(axis=1) * 0.85, axis=0).sum(axis=1)
        out = out.merge(deficit.rename("var_deficit_years").reset_index(),
                        on=config.KEY, how="left")

    out["n_weather_years"] = panel["weather_year"].nunique()
    return out


@lru_cache(maxsize=1)
def shc_trend() -> pd.DataFrame:
    """Direction and size of change in each soil component across SHC cycles.

    Soil fertility moves slowly, so these are small numbers — but a taluka
    whose organic carbon is falling is a different management problem from one
    where it is rising, and a single cycle cannot express that at all.
    """
    frames = []
    for i, cycle in enumerate(config.SHC_CYCLES):
        f = soil_health.build(cycle=cycle)
        f["cycle_index"] = i
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    cols = [c for c in TREND_VARS if c in panel.columns]
    rows = []
    for keys, g in panel.groupby(config.KEY, sort=True):
        rec = dict(zip(config.KEY, keys))
        g = g.sort_values("cycle_index")
        rec["shc_cycles_observed"] = len(g)
        # a taluka first surveyed in the latest cycle has no trend to measure,
        # and saying so is better than imputing zero change
        rec["shc_first_cycle"] = g["Cycle"].iat[0]
        for c in cols:
            y = g[c].to_numpy(dtype=float)
            if len(y) >= 2 and np.isfinite(y).all():
                x = g["cycle_index"].to_numpy(dtype=float)
                rec[f"shcTrend_{c}"] = float(np.polyfit(x, y, 1)[0])
                rec[f"shcDelta_{c}"] = float(y[-1] - y[0])
            else:
                rec[f"shcTrend_{c}"] = np.nan
                rec[f"shcDelta_{c}"] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows)
