"""Data contracts from plan §2. These guard the joins that silently corrupt rows."""
import pandas as pd
import pytest

from src import config
from src.data.load import (
    load_apy,
    load_fertiliser,
    load_shc,
    load_soil_type,
    load_weather,
    taluka_universe,
)


def test_shc_has_351_talukas():
    assert len(load_shc()) == 351


def test_soil_type_has_358_talukas():
    assert len(load_soil_type()) == 358


def test_seven_urban_talukas_lack_shc():
    st, shc = load_soil_type(), load_shc()
    missing = set(map(tuple, st[config.KEY].values)) - set(map(tuple, shc[config.KEY].values))
    assert missing == config.URBAN_NO_SHC


def test_taluka_alone_is_not_a_key():
    """Six taluka names repeat across districts — joining on Taluka alone corrupts them."""
    shc = load_shc()
    dups = shc.groupby("Taluka")["District"].nunique()
    assert set(dups[dups > 1].index) == {
        "ASHTI", "KALAMB", "KARANJA", "KARJAT", "KHED", "MALEGAON",
    }


def test_district_taluka_pair_is_unique_everywhere():
    for df in (load_shc(), load_soil_type()):
        assert not df.duplicated(subset=config.KEY).any()


def test_districts_reconcile_across_all_files():
    d_shc = set(load_shc()["District"])
    assert set(load_apy()["District"]) == d_shc
    assert set(load_fertiliser()["District"]) == d_shc
    assert set(load_weather()["District"]) >= d_shc
    assert len(d_shc) == 34


def test_weather_is_complete():
    wx = load_weather()
    assert len(wx) == 130_670
    assert wx[["Max_temperature", "Min_temperature", "Humidity", "Rainfall"]].isna().sum().sum() == 0
    assert wx.groupby(config.KEY)["Date"].nunique().eq(365).all()


def test_apy_is_a_single_year_of_district_labels():
    """Plan §2 fact 1+2: effective n is 34 districts, and the year is 2022-23."""
    apy = load_apy()
    assert apy["Year"].nunique() == 1
    assert apy["District"].nunique() == 34
    assert apy["Crop"].nunique() == 25


def test_fertiliser_flagged_rows_are_excluded():
    clean, full = load_fertiliser(), load_fertiliser.__wrapped__(clean_only=False)
    assert len(full) == 41_070
    assert len(clean) == 33_169
    assert clean["Data_Flag"].isna().all()
    # every per-tree dose is gone, so per-hectare arithmetic is safe
    assert not clean["Unit"].str.contains("Tree|Plant", case=False, na=False).any()


def test_universe_is_the_351_shc_talukas_with_geometry():
    u = taluka_universe()
    assert len(u) == 351
    assert u[["Latitude", "Longitude"]].notna().all().all()
    assert not u["urban_no_shc"].any()
