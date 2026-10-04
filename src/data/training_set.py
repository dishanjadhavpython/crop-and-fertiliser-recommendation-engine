"""The supervised learning frame (plan §2 fact 1, §5.1, §5.3).

Features live at taluka grain (351 rows); labels live at district grain
(34 districts). The effective sample size for supervised learning is therefore
**34, not 1,000** — every modelling decision downstream is sized for that.

Taluka features are aggregated to district by a weighted mean, using the Soil
Health Card sample count as the weight: a taluka with 4,480 samples should
carry more of its district's signal than one with 200.

Two targets are constructed, and the choice between them is the single most
consequential decision in the project (plan §7.3):

``yield_z``
    Within-crop z-scored yield. Never raw yield — sugarcane at 74 t/ha against
    sesamum at 0.27 t/ha otherwise dominates every split and produces the fake
    R2 = 0.928 the plan opens with.

``relevance``
    Revealed farmer preference, graded within district from area share. The
    area a district devotes to a crop is a strong, free signal of suitability.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_apy, load_apy_panel
from src.features.build_store import build_feature_store, feature_columns


@lru_cache(maxsize=64)
def district_features(with_context_z: bool = False, with_spatial: bool = False,
                      years: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Aggregate the taluka feature store to district grain.

    Weighted by SHC sample count. Also carries ``n_talukas`` and the summed
    sample count, both of which are legitimate confidence signals. ``years``
    is passed straight through to the store, so a caller can ask for the
    district as it looked **as of** a given crop year.
    """
    store = build_feature_store(with_context_z, with_spatial, years=years)
    cols = feature_columns(store)
    w = store["n_samples"].to_numpy(dtype=float)

    rows = []
    for district, g in store.groupby("District", sort=True):
        gw = g["n_samples"].to_numpy(dtype=float)
        gw = gw / gw.sum() if gw.sum() > 0 else np.full(len(g), 1 / len(g))
        rec = {"District": district,
               "n_talukas": len(g),
               "shc_samples_total": float(g["n_samples"].sum())}
        for c in cols:
            rec[c] = float(np.average(g[c].to_numpy(dtype=float), weights=gw))
        rows.append(rec)
    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def build(with_context_z: bool = False, with_spatial: bool = False,
          multi_year: bool = True) -> pd.DataFrame:
    """District x year x crop x season modelling frame with both targets.

    With ``multi_year`` the panel spans 2015-16..2022-23: **272 district-years
    instead of 34**. That is the single change the original plan identified as
    worth more than every modelling decision combined.
    """
    apy = load_apy_panel() if multi_year else load_apy()
    apy = apy[apy["Area"] > 0].copy()

    # ---- revealed preference: share of the district-season-year cropped area --
    # The query is (district, season, year): a farmer choosing in 2018 competes
    # against 2018's alternatives, not against a pooled decade.
    q = ["District", "Season", "Year"]
    apy["area_share"] = apy["Area"] / apy.groupby(q)["Area"].transform("sum")
    apy["area_share_district"] = (
        apy["Area"] / apy.groupby(["District", "Year"])["Area"].transform("sum"))

    # Relevance grades 1-4 are formed *within* each query group. Grading
    # globally would leak the district's overall scale into the label.
    apy["relevance"] = apy.groupby(q)["area_share"].transform(_grade).astype(int) + 1

    # Crops the district did *not* plant are the negatives, and without them
    # the task is not recommendation at all — it is re-ordering a list the
    # district has already chosen. Every crop grown in that season anywhere in
    # Maharashtra becomes a candidate, at relevance 0.
    apy = _add_unplanted_candidates(apy)

    # ---- yield target: within-crop z-score, never raw yield ------------------
    # Standardised within crop AND year: technology and prices drift over eight
    # years, so a 2015 tonne is not a 2022 tonne. Pooling the years would let
    # the model read that drift as agronomy.
    grp = apy.groupby(["Crop", "Year"])["Yield"]
    crop_mean, crop_std = grp.transform("mean"), grp.transform("std")
    apy["yield_z"] = ((apy["Yield"] - crop_mean) / crop_std.replace(0, np.nan)).fillna(0.0)
    apy["crop_yield_mean"] = crop_mean
    apy["crop_yield_std"] = crop_std

    # Each crop year is joined to the district as it could have been known THEN:
    # normals over the weather years that had already happened, plus — kept
    # separate, and kept away from the ranker — the weather that year actually
    # had. Before this, every row carried normals averaged over all 29 weather
    # years, so a 2015 row knew 2022's monsoon.
    frames = []
    for year, rows in apy.groupby("Year", sort=True):
        feats = district_features(with_context_z, with_spatial,
                                  years=_asof_weather_years(year))
        part = rows.merge(feats, on="District", how="inner", validate="m:1")
        matched = district_year_weather(year)
        if matched is not None:
            part = part.merge(matched, on="District", how="left", validate="m:1")
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    # Yield targets exist only for crops the district actually planted; the
    # S3 model trains on that subset, the ranker on the full candidate grid.
    df["has_yield"] = df["Yield"].notna() & (df["planted"] == 1)

    # crop and season identity, as the ablation's "identity only" baseline needs
    df["crop_id"] = df["Crop"].astype("category").cat.codes
    df["season_id"] = df["Season"].astype("category").cat.codes
    # a plain ordinal so the model can pick up a technology trend; trees cannot
    # extrapolate it, which is the honest behaviour under the temporal split
    df["year_index"] = df["Year"].map(
        {y: i for i, y in enumerate(sorted(df["Year"].unique()))}).astype(int)
    return df.reset_index(drop=True)


def _asof_weather_years(crop_year: str) -> tuple[str, ...]:
    """The weather years a question asked in ``crop_year`` could already know.

    A farmer choosing what to sow in 2015 knows every monsoon up to 2014 and
    none after it. Strictly earlier, never the year itself: the season's own
    weather is the thing being decided under, not an input to the decision.
    """
    cutoff = config.APY_WEATHER_MATCH.get(crop_year) or f"{crop_year[:4]}-{crop_year[7:9]}"
    earlier = tuple(y for y in config.WEATHER_YEARS if y < cutoff)
    # a crop year older than every weather year still needs a normal; the
    # earliest available year is the least wrong answer, and it is flagged by
    # n_weather_years in the store
    return earlier or (config.WEATHER_YEARS[0],)


@lru_cache(maxsize=32)
def district_year_weather(crop_year: str) -> pd.DataFrame | None:
    """The weather that crop year actually had, at district grain, prefixed ``wx_``.

    Legitimate for the **yield** model, which explains a season after it
    happened. Never for the ranker, which answers before it: ``model_features``
    excludes these columns unless asked, and a test asserts it.
    """
    weather_year = config.APY_WEATHER_MATCH.get(crop_year)
    if weather_year is None:
        return None

    from src.features import agroclimate

    taluka = agroclimate.build(weather_year)
    store = build_feature_store()[config.KEY + ["n_samples"]]
    merged = taluka.merge(store, on=config.KEY, how="inner", validate="1:1")
    value_cols = [c for c in merged.columns
                  if c not in config.KEY + ["n_samples", "weather_year"]
                  and pd.api.types.is_numeric_dtype(merged[c])]

    rows = []
    for district, g in merged.groupby("District", sort=True):
        w = g["n_samples"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(g), 1 / len(g))
        rec = {"District": district}
        for c in value_cols:
            rec[f"wx_{c}"] = float(np.average(g[c].to_numpy(dtype=float), weights=w))
        rows.append(rec)
    return pd.DataFrame(rows)


def _add_unplanted_candidates(apy: pd.DataFrame) -> pd.DataFrame:
    """Complete each (district, season) query with its unplanted candidates.

    The candidate pool is every crop in the APY vocabulary, for every season —
    not just the crops recorded in that season. A farmer asking "what should I
    grow in Rabi" is choosing from everything, so the ranker has to learn
    seasonality rather than being handed it. It also makes the task hard enough
    for the baselines to separate: with a season-filtered pool a random ranker
    already scores NDCG@5 = 0.49.
    """
    crops = sorted(apy["Crop"].unique())
    seasons = sorted(apy["Season"].unique())
    districts = sorted(apy["District"].unique())
    years = sorted(apy["Year"].unique())

    grid = pd.DataFrame(
        [(d, s, c, y) for y in years for s in seasons
         for d in districts for c in crops],
        columns=["District", "Season", "Crop", "Year"],
    )
    full = grid.merge(apy, on=["District", "Season", "Crop", "Year"], how="left")

    unplanted = full["Area"].isna()
    full.loc[unplanted, ["Area", "Production", "area_share",
                         "area_share_district", "relevance"]] = 0.0
    full["planted"] = (~unplanted).astype(int)

    # carry the district-constant and crop-constant columns onto the new rows
    for col in ("State", "Area_unit", "Production_unit", "Yield_unit"):
        if col in full.columns:
            full[col] = full[col].ffill().bfill()
    full["relevance"] = full["relevance"].astype(int)
    return full.reset_index(drop=True)


def _grade(share: pd.Series) -> pd.Series:
    """Relevance 0-4 from area share, ranked within the group.

    ``qcut`` on the rank rather than on the raw share: shares are extremely
    skewed (sugarcane can be 60% of a district) and quantiles of the raw value
    would put almost everything in one bucket.
    """
    n_grades = config.RANKER_RELEVANCE_GRADES - 1     # 0 is reserved for unplanted
    if len(share) == 1:
        return pd.Series([n_grades - 1], index=share.index)
    ranks = share.rank(method="first")
    try:
        return pd.qcut(ranks, min(n_grades, len(share)), labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(np.zeros(len(share), dtype=int), index=share.index)


def target_columns() -> list[str]:
    return ["Yield", "yield_z", "relevance", "area_share", "area_share_district"]


def model_features(df: pd.DataFrame, include_identity: bool = True,
                   include_year_weather: bool = False) -> list[str]:
    """Feature columns for the learned models, labels and keys excluded.

    ``wx_*`` — the crop year's own weather — is excluded by default. It is not
    a label, but for the ranker it is just as unusable: a farmer choosing a crop
    in June cannot know what that season's rainfall will be. Only the yield
    model, which explains a season afterwards, asks for it.
    """
    leaky = {
        "Area", "Production", "Yield", "yield_z", "relevance",
        "area_share", "area_share_district", "crop_yield_mean", "crop_yield_std",
        "planted", "has_yield",
        "Soil_Type_Share_pct", "Soil_Points_Sampled",
    }
    keys = {"State", "District", "Year", "Crop", "Season",
            "Area_unit", "Production_unit", "Yield_unit",
            "Cycle", "weather_year", "shc_first_cycle"}
    cols = [
        c for c in df.columns
        if c not in leaky | keys and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not include_year_weather:
        cols = [c for c in cols if not c.startswith("wx_")]
    if not include_identity:
        cols = [c for c in cols if c not in ("crop_id", "season_id")]
    return cols
