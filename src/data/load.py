"""Raw-file loaders with the data contracts from plan §2 and §9 week 1.

Every loader enforces the same three rules:

1. Keys are canonicalised to upper-cased, whitespace-trimmed ``(District, Taluka)``.
   Six taluka names repeat across districts, so the pair is the only safe key.
2. The seven urban talukas with no Soil Health Card record are flagged, never imputed.
3. Fertiliser rows are filtered on ``Data_Flag.isna()`` before any per-hectare
   arithmetic; the flagged rows carry per-tree doses and implausible quantities.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from src import config
from src.data.admin_changes import (
    NON_AGRICULTURAL_DISTRICTS,
    canonicalise_districts,
)


def canonicalise(df: pd.DataFrame, cols=("State", "District", "Taluka")) -> pd.DataFrame:
    """Upper-case and trim the key columns present in ``df``."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip().str.upper()
    return df


def _read(path) -> pd.DataFrame:
    # utf-8-sig: every raw CSV carries a BOM on its first header cell.
    return pd.read_csv(path, encoding="utf-8-sig")


@lru_cache(maxsize=None)
def load_shc(cycle: str | None = None) -> pd.DataFrame:
    """Soil Health Card parameters for one cycle (default: the reference cycle)."""
    cycle = cycle or config.SHC_REFERENCE_CYCLE
    df = canonicalise(_read(config.F_SHC_BY_CYCLE[cycle]))
    df = canonicalise_districts(df)
    df["Cycle"] = cycle
    return df


@lru_cache(maxsize=None)
def load_shc_cycles() -> dict:
    """Every Soil Health Card cycle, keyed by cycle label.

    Coverage grows 347 -> 350 -> 351 talukas across the three cycles. Those
    four talukas were newly *surveyed*, not newly created — see
    src/data/admin_changes.py.
    """
    return {c: load_shc(c) for c in config.SHC_CYCLES}


@lru_cache(maxsize=None)
def load_shc_panel() -> pd.DataFrame:
    """All cycles stacked, one row per (District, Taluka, Cycle)."""
    return pd.concat(load_shc_cycles().values(), ignore_index=True)


@lru_cache(maxsize=None)
def load_soil_type() -> pd.DataFrame:
    """Soil class/texture/depth/drainage — 358 talukas, includes the 7 urban ones."""
    df = canonicalise_districts(canonicalise(_read(config.F_SOIL_TYPE)))
    df["urban_no_shc"] = [
        (d, t) in config.URBAN_NO_SHC for d, t in zip(df["District"], df["Taluka"])
    ]
    return df


@lru_cache(maxsize=None)
def load_weather(year: str | None = None) -> pd.DataFrame:
    """Daily weather for one agricultural year (April -> March).

    358 talukas x 365 days, or 366 in the 2023-24 leap year.
    """
    year = year or "2025-26"
    df = canonicalise(_read(config.F_WEATHER_BY_YEAR[year]))
    df = canonicalise_districts(df)
    df["Date"] = pd.to_datetime(df["Date"])
    df["weather_year"] = year
    return df


@lru_cache(maxsize=None)
def load_weather_years() -> dict:
    """Every delivered weather year, keyed by label.

    Four years replaces the single year the original plan was sized for. That
    is still not a 30-year normal, but it is enough to characterise
    inter-annual variability, which one year cannot do at all.
    """
    return {y: load_weather(y) for y in config.WEATHER_YEARS}


@lru_cache(maxsize=None)
def load_apy(year: str | None = None) -> pd.DataFrame:
    """District crop statistics for one year — the label source."""
    year = year or "2022-2023"
    df = canonicalise(_read(config.F_APY_BY_YEAR[year]), cols=("State", "District"))
    df = canonicalise_districts(df)
    df["Crop"] = df["Crop"].astype("string").str.strip()
    df["Season"] = df["Season"].astype("string").str.strip()
    df["Year"] = year
    return df


@lru_cache(maxsize=None)
def load_apy_panel(drop_non_agricultural: bool = True) -> pd.DataFrame:
    """All eight APY years stacked — 34 district labels become 272 district-years.

    District names are canonicalised on the way in, so a historic spelling in
    an externally sourced file cannot silently create a 35th district. Mumbai
    Suburban, which reports agriculture in only two of the eight years and has
    no Soil Health Card record, is dropped by default.
    """
    df = pd.concat([load_apy(y) for y in config.APY_YEARS], ignore_index=True)
    if drop_non_agricultural:
        df = df[~df["District"].isin(NON_AGRICULTURAL_DISTRICTS)].copy()
    return df.reset_index(drop=True)


@lru_cache(maxsize=None)
def load_fertiliser(clean_only: bool = True) -> pd.DataFrame:
    """Fertiliser recommendation table.

    With ``clean_only`` (the default) the 7,901 rows carrying a ``Data_Flag``
    are dropped: 7,729 are per-tree doses that cannot be converted to
    per-hectare and ~144 are implausible (up to 3,990 kg/ha).
    """
    df = canonicalise(_read(config.F_FERT), cols=("State", "District"))
    df["Crop"] = df["Crop"].astype("string").str.strip()
    if clean_only:
        df = df[df["Data_Flag"].isna()].copy()
    return df.reset_index(drop=True)


@lru_cache(maxsize=None)
def load_coords() -> pd.DataFrame:
    """Taluka centroids, keyed the same way as every other table."""
    raw = json.loads(config.F_COORDS.read_text())
    rows = []
    for key, (lat, lon) in raw.items():
        state, district, taluka = key.split("|")
        rows.append(
            {
                "State": state.strip().upper(),
                "District": district.strip().upper(),
                "Taluka": taluka.strip().upper(),
                "Latitude": lat,
                "Longitude": lon,
            }
        )
    return pd.DataFrame(rows)


def taluka_universe() -> pd.DataFrame:
    """The 351 talukas that carry a Soil Health Card record — the modelling universe.

    Soil type, coordinates and the urban flag are joined on ``(District, Taluka)``.
    """
    shc = load_shc()[config.KEY].drop_duplicates()
    st = load_soil_type()
    coords = load_coords()[config.KEY + ["Latitude", "Longitude"]]

    out = shc.merge(st.drop(columns=["State"]), on=config.KEY, how="left", validate="1:1")
    out = out.merge(coords, on=config.KEY, how="left", validate="1:1", suffixes=("", "_c"))
    # soil_type already carries Latitude/Longitude; fall back to the JSON where absent
    for c in ("Latitude", "Longitude"):
        if f"{c}_c" in out.columns:
            out[c] = out[c].fillna(out[f"{c}_c"])
            out = out.drop(columns=[f"{c}_c"])
    return out.sort_values(config.KEY).reset_index(drop=True)
