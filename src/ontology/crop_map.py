"""APY <-> fertiliser-table crop vocabulary bridge (plan §6.6).

The two tables name crops in incompatible vocabularies: APY uses Indian-English
census style (*Bajra, Jowar, Arhar/Tur*), the fertiliser table uses agronomic
English (*Pearl Millet, Sorghum, Pigeon Pea*). Only 8 of 25 match on a raw
string join, so without this map the pipeline breaks between S1 and S4 for
two-thirds of crops.

Note on the plan text: §6.6 spells the targets ``Pearl millet``, ``Pigeon pea``,
``Finger millet`` and ``Tetraploid cotton``. The table actually uses title case,
so resolution here is case-insensitive and the map is unit-tested against the
real 78-crop list.
"""
from __future__ import annotations

from functools import lru_cache

from src.data.load import load_apy_panel, load_fertiliser

#: APY crop name -> fertiliser-table crop name.
CROP_ONTOLOGY: dict[str, str] = {
    # translated names
    "Arhar/Tur": "Pigeon Pea",
    "Bajra": "Pearl Millet",
    "Gram": "Chickpea",
    "Jowar": "Sorghum",
    "Ragi": "Finger Millet",
    "Sesamum": "Sesame",
    "Soyabean": "Soybean",
    "Urad": "Urdbean",
    "Moong(Green Gram)": "Mungbean",
    "Cotton(lint)": "Tetraploid Cotton",
    # exact matches
    "Rice": "Rice",
    "Wheat": "Wheat",
    "Maize": "Maize",
    "Groundnut": "Groundnut",
    "Sugarcane": "Sugarcane",
    "Sunflower": "Sunflower",
    "Safflower": "Safflower",
    "Linseed": "Linseed",
    # present only in the earlier APY years (2016-17 .. 2019-20); the
    # fertiliser table calls it Indian Mustard
    "Rapeseed &Mustard": "Indian Mustard",
}

#: APY crops with no fertiliser recipe. Served from the S2 rule scorer only.
#: Together these are ~1.5% of Maharashtra's cropped area.
UNMAPPED_APY_CROPS: frozenset[str] = frozenset(
    {
        "Niger seed",
        "Tobacco",
        # appears 2015-16 .. 2018-19 only; no castor recipe exists in the
        # fertiliser table, so it is served from the rule scorer alone
        "Castor seed",
        "Other Cereals",
        "Other Kharif pulses",
        "Other Rabi pulses",
        "Other Summer Pulses",
        "other oilseeds",
        # an aggregate of kodo, kutki, foxtail and barnyard millet; no single
        # recipe or envelope can represent the bucket
        "Small millets",
    }
)


@lru_cache(maxsize=None)
def _fert_crop_lookup() -> dict[str, str]:
    """Case-folded fertiliser crop name -> the table's own spelling."""
    return {c.casefold(): c for c in load_fertiliser()["Crop"].dropna().unique()}


def to_fertiliser_crop(apy_crop: str) -> str | None:
    """Resolve an APY crop name to the fertiliser table's spelling.

    Returns ``None`` for the seven aggregate/unmapped crops, which have no
    recipe and must fall through to the rule scorer.
    """
    target = CROP_ONTOLOGY.get(apy_crop)
    if target is None:
        return None
    return _fert_crop_lookup().get(target.casefold())


@lru_cache(maxsize=None)
def local_names() -> dict[str, str]:
    """Fertiliser crop name -> Devanagari local name, for Marathi output."""
    fert = load_fertiliser()[["Crop", "Crop_Local_Name"]].dropna().drop_duplicates("Crop")
    return dict(zip(fert["Crop"], fert["Crop_Local_Name"]))


def coverage_report() -> dict:
    """Mapped-crop count and the share of cropped area the map covers."""
    apy = load_apy_panel()
    apy = apy.assign(mapped=apy["Crop"].map(lambda c: to_fertiliser_crop(c) is not None))
    total_area = apy["Area"].sum()
    return {
        "apy_crops": int(apy["Crop"].nunique()),
        "mapped_crops": int(apy.loc[apy["mapped"], "Crop"].nunique()),
        "unmapped_crops": sorted(apy.loc[~apy["mapped"], "Crop"].unique()),
        "area_coverage_pct": round(100 * apy.loc[apy["mapped"], "Area"].sum() / total_area, 2),
    }
