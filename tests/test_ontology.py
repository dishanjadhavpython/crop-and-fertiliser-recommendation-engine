"""Crop ontology (plan §6.6) — a blocker between S1 and S4."""
from src.data.load import load_apy, load_apy_panel, load_fertiliser
from src.ontology.crop_map import (
    CROP_ONTOLOGY,
    UNMAPPED_APY_CROPS,
    coverage_report,
    local_names,
    to_fertiliser_crop,
)


def test_every_mapped_target_exists_in_the_fertiliser_table():
    for apy_name in CROP_ONTOLOGY:
        assert to_fertiliser_crop(apy_name) is not None, apy_name


def test_map_keys_are_real_apy_crops():
    """Every crop across all eight years is either mapped or explicitly unmapped.

    The panel carries 28 crops, three more than the single 2022-23 year:
    Rapeseed & Mustard and Castor seed appear only in the earlier years, and
    Small millets in a subset.
    """
    apy_crops = set(load_apy_panel()["Crop"])
    assert len(apy_crops) == 28
    assert set(CROP_ONTOLOGY) <= apy_crops
    assert UNMAPPED_APY_CROPS <= apy_crops
    assert set(CROP_ONTOLOGY) | UNMAPPED_APY_CROPS == apy_crops


def test_recovers_19_crops_and_985_pct_of_area():
    """Rapeseed & Mustard maps to the table's Indian Mustard, taking 18 to 19."""
    rep = coverage_report()
    assert rep["mapped_crops"] == 19
    assert rep["area_coverage_pct"] >= 98.4


def test_raw_string_join_would_only_match_8():
    """The number the plan cites as the reason this module exists."""
    fert = {c.casefold() for c in load_fertiliser()["Crop"]}
    naive = {c for c in load_apy()["Crop"] if c.casefold() in fert}
    assert len(naive) == 8


def test_unmapped_crops_resolve_to_none():
    for crop in UNMAPPED_APY_CROPS:
        assert to_fertiliser_crop(crop) is None


def test_marathi_names_available_for_mapped_crops():
    names = local_names()
    for apy_name in CROP_ONTOLOGY:
        assert names.get(to_fertiliser_crop(apy_name))
