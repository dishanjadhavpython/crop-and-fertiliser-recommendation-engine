"""Administrative geography over time — renames, splits and coverage onset.

Multi-year panel data on Indian districts is dangerous precisely because the
districts themselves are not constant. A join that ignores this silently
compares a 2015 district with a differently-shaped 2022 one, or drops a
district entirely because it was spelled differently that year.

Everything here was verified against the delivered files, not assumed. Where
the files already handle something, that is recorded as verified rather than
re-applied — applying a rename twice is its own bug.

--------------------------------------------------------------------------
WHAT CHANGED IN MAHARASHTRA, AND WHAT IT MEANS FOR THIS PANEL (2015-16..2022-23)
--------------------------------------------------------------------------

1. THREE DISTRICT RENAMES (2023). No boundary change, name only:
       Ahmednagar  -> Ahilyanagar
       Osmanabad   -> Dharashiv
       Aurangabad  -> Chhatrapati Sambhajinagar
   VERIFIED: the delivered APY files already use the post-2023 names in every
   year including 2015-16, so they are retro-harmonised. The mapping is kept
   here so that any *externally* sourced file can be normalised on the way in,
   and so the rename is documented rather than invisible.

2. PALGHAR DISTRICT, created 1 August 2014 by splitting Thane.
   VERIFIED: Palghar appears as its own district in every delivered year from
   2015-16 on, and Thane appears alongside it. The panel therefore begins
   *after* the split and needs no reconstruction. This is the single most
   important thing to have checked: had the panel started before 2014-15,
   Thane's area and production would jump discontinuously and every model
   would read that administrative event as an agronomic one.

3. MUMBAI SUBURBAN reports agriculture only intermittently (2016-17 and
   2019-20 of the eight years). It has no Soil Health Card record at all and
   is already excluded from the modelling universe as an urban district.

4. SOIL HEALTH CARD COVERAGE EXPANDED, but no taluka was created.
   The 2023-24 cycle covers 347 talukas, 2024-25 covers 350, 2025-26 covers
   351. The four that appear late are Himayatnagar and Mudkhed (Nanded), and
   Ghatanji and Kelapur (Yavatmal).
   VERIFIED: all four are present in the static soil-type file and in the
   2022-23 daily weather, so they existed throughout — they were simply not
   sampled in the earlier cycle. Their sample counts (768-1,409) sit well
   below the state median of 2,380, consistent with newer partial surveys.
   The correct treatment is therefore "no measurement yet", NOT "did not
   exist": the taluka is flagged, its onset cycle recorded, and the existing
   sample-count weighting already discounts it.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

#: Historic district name -> the name used in this repository.
#: Applied to any externally sourced file; the delivered files already comply.
DISTRICT_RENAMES: dict[str, str] = {
    "AHMEDNAGAR": "AHILYANAGAR",
    "AHMADNAGAR": "AHILYANAGAR",
    "OSMANABAD": "DHARASHIV",
    "USMANABAD": "DHARASHIV",
    "AURANGABAD": "CHHATRAPATI SAMBHAJINAGAR",
    "SAMBHAJINAGAR": "CHHATRAPATI SAMBHAJINAGAR",
    "GONDIYA": "GONDIA",
    "BULDANA": "BULDHANA",
    "BID": "BEED",
}

#: District splits: child -> (parent, first agricultural year the child reports
#: separately). Rows for the child before that year live inside the parent.
DISTRICT_SPLITS: dict[str, tuple[str, str]] = {
    "PALGHAR": ("THANE", "2014-2015"),      # created 1 Aug 2014
}

#: Districts excluded from the modelling universe, with the reason.
NON_AGRICULTURAL_DISTRICTS: dict[str, str] = {
    "MUMBAI SUBURBAN": "urban; reports agriculture intermittently and has no "
                       "Soil Health Card record",
    "MUMBAI CITY": "urban; no agricultural reporting",
}

#: Talukas whose Soil Health Card coverage begins after the first cycle.
#: These are coverage gaps, not new administrative units — see the module
#: docstring for the verification.
LATE_SHC_COVERAGE: dict[tuple[str, str], str] = {
    ("NANDED", "HIMAYATNAGAR"): "2024-25",
    ("NANDED", "MUDKHED"): "2024-25",
    ("YAVATMAL", "GHATANJI"): "2024-25",
    ("YAVATMAL", "KELAPUR"): "2025-26",
}


def canonical_district(name: str) -> str:
    """Normalise any historic or variant district spelling to the current one."""
    key = str(name).strip().upper()
    return DISTRICT_RENAMES.get(key, key)


def canonicalise_districts(df: pd.DataFrame, col: str = "District") -> pd.DataFrame:
    """Apply the rename map to a frame's district column."""
    out = df.copy()
    out[col] = out[col].map(canonical_district)
    return out


def split_is_resolved(district: str, year: str) -> bool:
    """True when ``district`` reports separately in ``year``.

    A child district before its split year is not missing data — its area sits
    inside the parent, and treating the two as one series is the only correct
    comparison.
    """
    parent_year = DISTRICT_SPLITS.get(canonical_district(district))
    if parent_year is None:
        return True
    return year >= parent_year[1]


@lru_cache(maxsize=None)
def shc_coverage_onset() -> dict[tuple[str, str], str]:
    """Earliest Soil Health Card cycle in which each taluka appears.

    Derived from the delivered files rather than declared, so it stays correct
    when a new cycle is added.
    """
    from src.data.load import load_shc_cycles

    onset: dict[tuple[str, str], str] = {}
    for cycle, df in sorted(load_shc_cycles().items()):
        for d, t in zip(df["District"], df["Taluka"]):
            onset.setdefault((d, t), cycle)
    return onset


def audit(apy: pd.DataFrame) -> pd.DataFrame:
    """Per-year audit of the district panel — the table to put in the report."""
    rows = []
    for year, g in apy.groupby("Year"):
        present = set(g["District"])
        rows.append({
            "Year": year,
            "districts": len(present),
            "has_palghar": "PALGHAR" in present,
            "has_thane": "THANE" in present,
            "urban_present": sorted(present & set(NON_AGRICULTURAL_DISTRICTS)),
            "crops": g["Crop"].nunique(),
        })
    return pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
