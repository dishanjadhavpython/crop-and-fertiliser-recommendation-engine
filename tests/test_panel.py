"""The multi-year panel: administrative geography and file integrity.

Districts are not constant over time, and neither is Soil Health Card
coverage. These tests pin down every claim made in
``src/data/admin_changes.py`` against the delivered files, so that a future
data drop cannot silently change the panel's shape.
"""
import hashlib
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import admin_changes as ac
from src.data.load import (
    load_apy,
    load_apy_panel,
    load_shc_cycles,
    load_weather,
    load_weather_years,
)


# ------------------------------------------------------- file integrity ----
@pytest.mark.parametrize("year", config.WEATHER_YEARS)
def test_weather_file_contents_match_its_filename(year):
    """A mislabelled weather file already slipped in once.

    The delivered ``2022-04-01_to_2023-03-31`` file was a byte-identical copy
    of the 2025-26 file, and its own Date column proved it. Had it been used,
    it would have manufactured a false claim that weather is contemporaneous
    with the 2022-23 crop labels.
    """
    wx = load_weather(year)
    lo, hi = config.WEATHER_YEAR_SPANS[year]
    assert str(wx["Date"].min().date()) == lo
    assert str(wx["Date"].max().date()) == hi


def test_weather_years_are_distinct_files():
    digests = {}
    for year, path in config.F_WEATHER_BY_YEAR.items():
        digests[year] = hashlib.md5(path.read_bytes()).hexdigest()
    assert len(set(digests.values())) == len(digests), f"duplicate weather files: {digests}"


def test_the_rejected_duplicate_is_quarantined_and_the_real_year_took_its_name():
    """The delivered 2022-23 file was a copy of the 2025-26 one.

    That name is now referenced, because the genuine 2022-23 weather occupies
    it — so the check that matters is no longer "is this name absent" but
    "is the copy out of reach, and does the file behind the name say 2022-23".
    The copy is kept rather than deleted: what was delivered is part of the
    record.
    """
    rejected = (config.RAW / "_rejected"
                / "maharashtra_daily_weather_taluka_2022-04-01_to_2023-03-31.csv")
    assert rejected.exists(), "the rejected duplicate should be kept, not deleted"
    assert rejected not in config.F_WEATHER_BY_YEAR.values()

    wx = load_weather("2022-23")
    assert str(wx["Date"].min().date()) == "2022-04-01"
    assert str(wx["Date"].max().date()) == "2023-03-31"


def test_weather_covers_every_day_including_every_leap_year():
    """Six of the 29 agricultural years contain a 29 February, not just one."""
    for year, wx in load_weather_years().items():
        lo, hi = (date.fromisoformat(d) for d in config.WEATHER_YEAR_SPANS[year])
        expected = (hi - lo).days + 1
        days = wx.groupby(config.KEY)["Date"].nunique()
        assert days.eq(expected).all(), f"{year}: expected {expected} days"


# ------------------------------------------------------------- the panel ----
def test_panel_spans_eight_years_and_272_district_years():
    p = load_apy_panel()
    assert p["Year"].nunique() == 8
    assert p["District"].nunique() == 34
    assert p.groupby(["District", "Year"]).ngroups == 272
    assert len(p) == 7035


def test_every_year_has_the_same_district_set():
    """A district appearing in some years and not others distorts every trend."""
    p = load_apy_panel()
    per_year = p.groupby("Year")["District"].apply(frozenset)
    assert per_year.nunique() == 1, "district set varies across years"


# ----------------------------------------------- administrative geography ---
def test_palghar_reports_separately_in_every_year():
    """Created 1 Aug 2014 from Thane. Had the panel started earlier, Thane's
    area would drop discontinuously and the model would read an administrative
    event as an agronomic one."""
    p = load_apy_panel()
    for year, g in p.groupby("Year"):
        assert "PALGHAR" in set(g["District"]), year
        assert "THANE" in set(g["District"]), year


def test_renamed_districts_use_the_current_name_everywhere():
    p = load_apy_panel()
    names = set(p["District"])
    for historic in ("AHMEDNAGAR", "OSMANABAD", "AURANGABAD"):
        assert historic not in names
    for current in ("AHILYANAGAR", "DHARASHIV", "CHHATRAPATI SAMBHAJINAGAR"):
        assert current in names


def test_rename_map_normalises_historic_spellings():
    assert ac.canonical_district("Ahmednagar") == "AHILYANAGAR"
    assert ac.canonical_district("osmanabad") == "DHARASHIV"
    assert ac.canonical_district(" Aurangabad ") == "CHHATRAPATI SAMBHAJINAGAR"
    # a current name must pass through untouched, or applying the map twice breaks it
    assert ac.canonical_district("AHILYANAGAR") == "AHILYANAGAR"
    assert ac.canonical_district("Kolhapur") == "KOLHAPUR"


def test_urban_districts_are_excluded_from_the_panel():
    p = load_apy_panel()
    assert not (set(p["District"]) & set(ac.NON_AGRICULTURAL_DISTRICTS))
    # ... and they really are present in the raw files, so the drop is doing work
    raw = pd.concat([load_apy(y) for y in config.APY_YEARS], ignore_index=True)
    assert "MUMBAI SUBURBAN" in set(raw["District"])


def test_split_resolution_knows_when_palghar_became_a_district():
    assert not ac.split_is_resolved("Palghar", "2013-2014")
    assert ac.split_is_resolved("Palghar", "2015-2016")
    assert ac.split_is_resolved("Kolhapur", "1997-1998")     # never split


# ---------------------------------------- Soil Health Card coverage onset ---
def test_shc_coverage_grows_but_no_taluka_is_created():
    """The four late talukas existed all along — they were newly surveyed.

    Treating a coverage gap as a creation event would teach the model that
    these places appeared from nowhere in 2024.
    """
    from src.data.load import load_soil_type

    cycles = load_shc_cycles()
    assert [len(c) for c in cycles.values()] == [347, 350, 351]

    first, last = cycles[config.SHC_CYCLES[0]], cycles[config.SHC_CYCLES[-1]]
    late = set(zip(last["District"], last["Taluka"])) - set(zip(first["District"], first["Taluka"]))
    assert late == set(ac.LATE_SHC_COVERAGE)

    # every one of them is in the static soil-type file, so it existed throughout
    existing = set(zip(load_soil_type()["District"], load_soil_type()["Taluka"]))
    assert late <= existing


def test_coverage_onset_is_derived_from_the_files_not_declared():
    onset = ac.shc_coverage_onset()
    assert len(onset) == 351
    for key, cycle in ac.LATE_SHC_COVERAGE.items():
        assert onset[key] == cycle
    assert sum(1 for v in onset.values() if v == config.SHC_CYCLES[0]) == 347


def test_late_surveyed_talukas_carry_fewer_samples():
    """Consistent with a newer partial survey, and already discounted by the
    sample-count weighting."""
    latest = load_shc_cycles()[config.SHC_CYCLES[-1]]
    late_names = {t for _, t in ac.LATE_SHC_COVERAGE}
    late = latest[latest["Taluka"].isin(late_names)]["Nitrogen_N_Samples"]
    assert len(late) == 4
    assert late.max() < latest["Nitrogen_N_Samples"].median()
