"""The weather panel's two honesty rules.

The backfill gave every crop year its own weather, and that creates exactly one
new way to cheat: letting a model see weather that had not happened when the
question was asked. Two rules keep it honest, and they are tested rather than
asserted in a comment.

    as-of normals   a crop year averages only EARLIER weather years
    wx_*            the season's own weather reaches the yield model, never the ranker
"""
import pandas as pd
import pytest

from src import config
from src.data.training_set import (
    _asof_weather_years,
    build,
    district_year_weather,
    model_features,
)

DF = build()


def test_every_crop_year_now_has_its_own_weather():
    """APY_WEATHER_MATCH was empty for the whole life of the project."""
    assert set(config.APY_WEATHER_MATCH) == set(config.APY_YEARS)
    for crop_year, weather_year in config.APY_WEATHER_MATCH.items():
        assert weather_year in config.F_WEATHER_BY_YEAR


def test_asof_normals_never_include_the_query_year_or_later():
    for crop_year, weather_year in config.APY_WEATHER_MATCH.items():
        asof = _asof_weather_years(crop_year)
        assert asof, f"{crop_year} has no earlier weather at all"
        assert all(y < weather_year for y in asof), \
            f"{crop_year}: as-of window reaches {max(asof)}, at or past {weather_year}"


def test_asof_windows_grow_with_the_panel():
    """A later crop year knows strictly more history than an earlier one."""
    years = sorted(config.APY_WEATHER_MATCH)
    sizes = [len(_asof_weather_years(y)) for y in years]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_year_matched_weather_is_kept_out_of_the_ranker():
    ranker_feats = model_features(DF)
    yield_feats = model_features(DF, include_year_weather=True)
    assert not [c for c in ranker_feats if c.startswith("wx_")], \
        "the season's own weather must never reach a model that answers before the season"
    assert [c for c in yield_feats if c.startswith("wx_")]


def test_year_matched_weather_actually_differs_between_years():
    """If wx_ columns were constant the join has silently gone wrong."""
    rain = DF.groupby("Year")["wx_rain_Kharif"].mean()
    assert rain.notna().all()
    assert rain.std() > 1.0, "Kharif rainfall identical across years — check the join"


def test_district_year_weather_is_one_row_per_district():
    wx = district_year_weather("2018-2019")
    assert wx is not None
    assert wx["District"].is_unique
    assert len(wx) == DF["District"].nunique()
