"""Assemble the taluka feature store — one deterministic command (plan §9 week 2).

    python -m src.features.build_store

Produces ``data/features/taluka_features.parquet``: 351 talukas x ~144 features,
reproducible from the raw files alone.
"""
from __future__ import annotations

import argparse
from functools import lru_cache

import pandas as pd

from src import config
from src.features import (agroclimate, climatology, interactions,
                          soil_health, soil_physical)


@lru_cache(maxsize=64)
def build_feature_store(with_context_z: bool = False, with_spatial: bool = False,
                        multi_year: bool = True,
                        years: tuple[str, ...] | None = None) -> pd.DataFrame:
    """The taluka feature store.

    With ``multi_year`` (the default) the agro-climatic block is the **normal**
    across the weather years rather than a single year's snapshot, and two
    blocks that a single year cannot produce are added: inter-annual
    variability, and the trend of each soil component across the Soil Health
    Card cycles.

    ``years`` restricts which weather years the normal is taken over. Serving
    leaves it unset — at serving time there is no future to leak, and "what is
    this place typically like" is the whole question. Evaluation passes the
    years available **as of** the crop year being scored.
    """
    a = soil_health.build(with_context_z=with_context_z)
    b = soil_physical.build()
    c = agroclimate.build()

    df = a.merge(b, on=config.KEY, how="inner", validate="1:1")
    df = df.merge(c, on=config.KEY, how="inner", validate="1:1")

    if multi_year:
        clim = climatology.climatology(years=years)
        df = df.merge(clim, on=config.KEY, how="left", validate="1:1")
        # The normal replaces the snapshot wherever both exist, so the
        # interaction block downstream reasons over the normal — and the twin is
        # then dropped. Keeping both shipped 81 exactly duplicated columns into
        # the model: 81 extra chances for LightGBM's column sampling to draw the
        # same signal twice, and 81 columns of noise in every importance table.
        twins = [c for c in df.columns
                 if c.startswith("clim_") and c[len("clim_"):] in df.columns]
        for twin in twins:
            base = twin[len("clim_"):]
            df[base] = df[twin].fillna(df[base])
        df = df.drop(columns=twins)
        df = df.merge(climatology.shc_trend(), on=config.KEY, how="left", validate="1:1")

    df = interactions.build(df, with_spatial=with_spatial)
    return df.sort_values(config.KEY).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric modelling columns — keys, labels and the categorical worst-micro excluded."""
    drop = set(config.KEY) | {"micro_def_worst", "Cycle", "weather_year",
                              "shc_first_cycle"}
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-context-z", action="store_true",
                   help="add contextual z-scores (hurt at n=34, see §7.1)")
    p.add_argument("--with-spatial", action="store_true",
                   help="add kNN spatial smoothing (hurt at n=34, see §7.1)")
    p.add_argument("--single-year", action="store_true",
                   help="use one weather year only, as the original plan did")
    args = p.parse_args()

    df = build_feature_store(args.with_context_z, args.with_spatial,
                             multi_year=not args.single_year)
    out = config.FEATURES / "taluka_features.parquet"
    df.to_parquet(out, index=False)

    total = len(feature_columns(df))
    blocks = {
        "A soil health": len([c for c in soil_health.build().columns
                              if c in df.columns and c not in config.KEY]),
        "B soil physical": len([c for c in soil_physical.build().columns
                                if c in df.columns and c not in config.KEY]),
        "C agro-climate (normals)": len([c for c in agroclimate.build().columns
                                         if c in df.columns and c not in config.KEY]),
        "E inter-annual variability": len([c for c in df.columns
                                           if c.startswith("var_")]),
        "F soil-health trend": len([c for c in df.columns
                                    if c.startswith(("shcTrend_", "shcDelta_"))]),
    }
    blocks["D interactions + raw climate"] = total - sum(blocks.values())
    print(f"feature store -> {out}")
    print(f"  {len(df)} talukas x {total} numeric features")
    for name, n in blocks.items():
        print(f"    {name:<18} {n:>4}")


if __name__ == "__main__":
    main()
