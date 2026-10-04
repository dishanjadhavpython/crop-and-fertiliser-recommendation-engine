"""Block C — agro-climatology (plan §4.3).

130,670 daily rows collapsed into agronomically meaningful seasonal summaries.
Daily data must never reach a model whose labels are annual.

Two caveats the report must carry (plan §2 facts 2 and 3): the weather year is
Apr 2025 - Mar 2026 while the yields are 2022-23, so these features are a
*climatology descriptor* — "what this place is typically like" — and never a
causal weather signal. With one year there is no inter-annual variation, so no
model here can learn a dry-year/wet-year response.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_weather, taluka_universe

SOLAR_CONSTANT = 0.0820          # MJ m-2 min-1
LATENT_HEAT = 2.45               # MJ kg-1, to convert MJ m-2 d-1 -> mm d-1


# --------------------------------------------------------------------------
# ET0 — Hargreaves. Solar radiation is unavailable, so Penman-Monteith is out;
# Hargreaves needs only Tmax, Tmin and latitude and is the FAO-sanctioned
# fallback. This unlocks the water balance, where the real signal lives.
# --------------------------------------------------------------------------
def extraterrestrial_radiation(latitude_deg, doy) -> np.ndarray:
    """FAO-56 eq. 21 — Ra in MJ m-2 day-1 from latitude and day-of-year."""
    phi = np.deg2rad(np.asarray(latitude_deg, dtype=float))
    doy = np.asarray(doy, dtype=float)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)              # eq. 23
    delta = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)        # eq. 24
    # sunset hour angle, clipped for polar cases that cannot arise here
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0))   # eq. 25
    return (24 * 60 / np.pi) * SOLAR_CONSTANT * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws)
    )


def hargreaves_et0(tmax, tmin, latitude_deg, doy) -> np.ndarray:
    """ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin) / 2.45  [mm/day]."""
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    ra = extraterrestrial_radiation(latitude_deg, doy)
    tmean = (tmax + tmin) / 2.0
    trange = np.clip(tmax - tmin, 0.0, None)   # guard against inverted records
    return 0.0023 * ra * (tmean + 17.8) * np.sqrt(trange) / LATENT_HEAT


# --------------------------------------------------------------------------
# Rainfall structure
# --------------------------------------------------------------------------
def longest_dry_spell(rain, threshold: float = config.DRY_DAY_MM) -> int:
    """Longest run of consecutive days below ``threshold`` mm.

    The drought indicator that matters: 1,200 mm delivered in four bursts is
    agriculturally very different from 1,200 mm spread evenly.
    """
    best = run = 0
    for r in np.asarray(rain, dtype=float):
        run = run + 1 if r < threshold else 0
        if run > best:
            best = run
    return best


def monsoon_onset_doy(dates: pd.Series, rain: pd.Series) -> float:
    """First day on/after 1 June whose 5-day forward rainfall reaches 25 mm.

    Onset date determines what can be sown at all. Returns NaN if the criterion
    is never met within the year.
    """
    s = pd.Series(np.asarray(rain, dtype=float), index=pd.DatetimeIndex(dates)).sort_index()
    window = config.MONSOON_ONSET_WINDOW_DAYS
    fwd = s.rolling(window).sum().shift(-(window - 1))
    after_june = fwd[fwd.index.month >= 6]
    hit = after_june[after_june >= config.MONSOON_ONSET_WINDOW_MM]
    if hit.empty:
        return float("nan")
    return float(hit.index[0].dayofyear)


def rain_concentration(rain) -> float:
    """Share of rainfall falling in the 5 wettest days.

    High concentration means runoff and erosion rather than stored soil moisture.
    """
    r = np.sort(np.asarray(rain, dtype=float))[::-1]
    total = r.sum()
    return float(r[:5].sum() / total) if total > 0 else 0.0


# --------------------------------------------------------------------------
def _season_of(month: pd.Series) -> pd.Series:
    lookup = {m: s for s, months in config.SEASON_MONTHS.items() for m in months}
    return month.map(lookup)


def _summarise(g: pd.DataFrame, tag: str) -> dict:
    """All per-window features for one taluka-season (or the whole year)."""
    rain = g["Rainfall"].to_numpy(dtype=float)
    tmax = g["Max_temperature"].to_numpy(dtype=float)
    tmin = g["Min_temperature"].to_numpy(dtype=float)
    tmean = (tmax + tmin) / 2.0
    rh = g["Humidity"].to_numpy(dtype=float)
    et0 = g["et0"].to_numpy(dtype=float)

    total_rain = float(rain.sum())
    total_et0 = float(et0.sum())
    ndays = len(g)

    f = {
        # --- water ---
        f"rain_{tag}": total_rain,
        f"rainy_days_{tag}": int((rain > config.DRY_DAY_MM).sum()),
        f"heavy_days_{tag}": int((rain > config.HEAVY_RAIN_MM).sum()),
        f"rain_cv_{tag}": float(rain.std(ddof=0) / rain.mean()) if rain.mean() > 0 else 0.0,
        f"dry_spell_{tag}": longest_dry_spell(rain),
        f"rain_conc5_{tag}": rain_concentration(rain),
        # --- atmospheric demand ---
        f"et0_{tag}": total_et0,
        # classic agro-climatic zone classifier
        f"aridity_{tag}": total_rain / total_et0 if total_et0 > 0 else np.nan,
        # negative Rabi CWB is literally the irrigation requirement
        f"cwb_{tag}": total_rain - total_et0,
        # FAO's length of growing period
        f"lgp_{tag}": int((rain > 0.5 * et0).sum()),
        # --- thermal ---
        f"gdd_{tag}": float(np.clip(tmean - config.GDD_BASE_C, 0, None).sum()),
        f"tmax_mean_{tag}": float(tmax.mean()),
        f"tmin_mean_{tag}": float(tmin.mean()),
        # drives sugar accumulation and grain filling
        f"dtr_{tag}": float((tmax - tmin).mean()),
        f"heat_days_{tag}": int((tmax > config.HEAT_STRESS_C).sum()),
        # matters for flowering-stage sterility
        f"severe_heat_days_{tag}": int((tmax > config.SEVERE_HEAT_C).sum()),
        # constrains Rabi crops
        f"cold_days_{tag}": int((tmin < config.COLD_STRESS_C).sum()),
        # --- biotic ---
        f"rh_mean_{tag}": float(rh.mean()),
        # the envelope for blast, blight and downy mildew
        f"fungal_days_{tag}": int(
            (
                (rh > config.FUNGAL_RH_PCT)
                & (tmax >= config.FUNGAL_TMAX_RANGE[0])
                & (tmax <= config.FUNGAL_TMAX_RANGE[1])
            ).sum()
        ),
        f"ndays_{tag}": ndays,
    }
    return f


@lru_cache(maxsize=None)   # 29 weather years now, not 3
def build(year: str | None = None) -> pd.DataFrame:
    """Return the Block C feature frame for one weather year, one row per taluka."""
    wx = load_weather(year)
    coords = taluka_universe()[config.KEY + ["Latitude"]]
    wx = wx.merge(coords, on=config.KEY, how="inner", validate="m:1")

    wx["doy"] = wx["Date"].dt.dayofyear
    wx["et0"] = hargreaves_et0(
        wx["Max_temperature"], wx["Min_temperature"], wx["Latitude"], wx["doy"]
    )
    wx["season"] = _season_of(wx["Date"].dt.month)

    rows = []
    for keys, g in wx.groupby(config.KEY, sort=True):
        g = g.sort_values("Date")
        rec = dict(zip(config.KEY, keys))
        rec.update(_summarise(g, "annual"))
        rec["monsoon_onset_doy"] = monsoon_onset_doy(g["Date"], g["Rainfall"])
        for season in config.SEASONS:
            sg = g[g["season"] == season]
            rec.update(_summarise(sg, season))
        rows.append(rec)

    out = pd.DataFrame(rows)
    # onset never met (arid talukas): fall back to the latest observed onset so
    # the feature reads as "very late" rather than as missing
    out["monsoon_onset_doy"] = out["monsoon_onset_doy"].fillna(out["monsoon_onset_doy"].max())
    out["weather_year"] = year or "2025-26"
    return out
