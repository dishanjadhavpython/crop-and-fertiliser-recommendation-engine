"""Data-quality report (plan §9 week 1 deliverable).

    python -m src.data.quality_report

Every structural claim in the project plan is re-checked against the raw files
and the result written to ``reports/data_quality.md``. Where this repository
found something the plan does not state, it is recorded under "Additional
findings" rather than quietly worked around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data.load import (
    load_apy,
    load_apy_panel,
    load_shc_cycles,
    load_weather_years,
    load_coords,
    load_fertiliser,
    load_shc,
    load_soil_type,
    load_weather,
    taluka_universe,
)
from src.ontology.crop_map import CROP_ONTOLOGY, coverage_report, to_fertiliser_crop
from src.rules import fertiliser as fz


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def verified_claims() -> pd.DataFrame:
    shc, st, wx = load_shc(), load_soil_type(), load_weather()
    apy = load_apy()
    full = load_fertiliser.__wrapped__(clean_only=False)
    clean = load_fertiliser()

    key = lambda df: set(zip(df["District"], df["Taluka"]))
    urban = key(st) - key(shc)
    dup = shc.groupby("Taluka")["District"].nunique()
    dup = set(dup[dup > 1].index)

    t = fz._table()
    amb = clean.groupby(
        ["District", "Crop", "Crop_Variety", "Crop_Irrigation", "Crop_Season",
         "Soil_Class", "Category", "Option", "Fertilizer"], dropna=False
    )["Quantity"].nunique()

    archetypes_ok = all(
        t[t["Soil_Class"] == cls][f"Soil_{n}_kg_per_ha" if n != "OC" else "Soil_OC_pct"]
        .unique().tolist() == [config.SOIL_ARCHETYPES[n][cls]]
        for cls in config.SOIL_CLASSES for n in ("N", "P", "K", "OC")
    )
    cov = coverage_report()

    rows = [
        ("Soil Health Card talukas", "351", f"{len(shc)}"),
        ("Soil-type talukas", "358", f"{len(st)}"),
        ("Urban talukas with no SHC record", "7", f"{len(urban)}"),
        ("Duplicated taluka names across districts", "6", f"{len(dup)}"),
        ("Districts (all five files agree)", "34", f"{apy['District'].nunique()}"),
        ("Daily weather rows", "130,670", f"{len(wx):,}"),
        ("Missing weather values", "0", f"{int(wx.isna().sum().sum())}"),
        ("APY rows / crops / years", "1,000 / 25 / 1", 
         f"{len(apy):,} / {apy['Crop'].nunique()} / {apy['Year'].nunique()}"),
        ("Fertiliser rows (all)", "41,070", f"{len(full):,}"),
        ("Fertiliser rows after Data_Flag filter", "33,169", f"{len(clean):,}"),
        ("Fertiliser keys with an ambiguous quantity", "0", f"{int((amb > 1).sum())}"),
        ("Soil-class archetypes are the SHC band midpoints", "yes",
         "yes" if archetypes_ok else "NO"),
        ("Crops recovered by the ontology", "18 of 25", f"{cov['mapped_crops']} of {cov['apy_crops']}"),
        ("Cropped area the ontology covers", ">= 98.5%", f"{cov['area_coverage_pct']}%"),
    ]
    df = pd.DataFrame(rows, columns=["Claim in the plan", "Stated", "Measured here"])

    def ok(stated: str, measured: str) -> str:
        stated, measured = stated.replace(",", ""), measured.replace(",", "")
        if stated.startswith(">="):
            return "yes" if float(measured.rstrip("%")) >= float(
                stated[2:].strip().rstrip("%")) else "check"
        return "yes" if stated == measured else "check"

    df["Verified"] = [ok(s, m) for s, m in zip(df["Stated"], df["Measured here"])]
    return df


def flag_breakdown() -> pd.DataFrame:
    full = load_fertiliser.__wrapped__(clean_only=False)
    f = full["Data_Flag"].fillna("(clean — usable per-hectare row)")
    grp = f.str.replace(r"\(\d[\d,.]*\s*kg/ha\)", "(...)", regex=True) \
           .str.replace(r"differ by [\d.]+ kg/ha", "differ by ... kg/ha", regex=True)
    out = grp.value_counts().reset_index()
    out.columns = ["Data_Flag", "rows"]
    out["share"] = (100 * out["rows"] / len(full)).round(2).astype(str) + "%"
    return out


def urban_talukas() -> pd.DataFrame:
    st, shc = load_soil_type(), load_shc()
    missing = set(zip(st["District"], st["Taluka"])) - set(zip(shc["District"], shc["Taluka"]))
    return pd.DataFrame(sorted(missing), columns=["District", "Taluka"]).assign(
        action="dropped from the modelling universe with an explicit flag; never imputed")


def duplicate_talukas() -> pd.DataFrame:
    shc = load_shc()
    dup = shc.groupby("Taluka")["District"].nunique()
    names = sorted(dup[dup > 1].index)
    return (shc[shc["Taluka"].isin(names)][["Taluka", "District"]]
            .sort_values(["Taluka", "District"]).reset_index(drop=True))


def ontology_table() -> pd.DataFrame:
    rows = [{"APY name": a, "Fertiliser-table name": to_fertiliser_crop(a)}
            for a in sorted(CROP_ONTOLOGY)]
    return pd.DataFrame(rows)


def fertiliser_coverage() -> pd.DataFrame:
    t = fz._table()
    have = set(zip(t["District"], t["Crop"]))
    districts = sorted(t["District"].unique())
    rows = []
    for apy in sorted(CROP_ONTOLOGY):
        fc = to_fertiliser_crop(apy)
        n = sum((d, fc) in have for d in districts)
        rows.append({"Crop": apy, "Districts with a published recipe": n,
                     "of": len(districts),
                     "coverage": f"{100 * n / len(districts):.0f}%"})
    return pd.DataFrame(rows).sort_values("Districts with a published recipe")


def additional_findings() -> list[str]:
    t = fz._table()
    have = set(zip(t["District"], t["Crop"]))
    districts = sorted(t["District"].unique())
    cells = sum((d, to_fertiliser_crop(a)) in have
                for a in CROP_ONTOLOGY for d in districts)
    total = len(CROP_ONTOLOGY) * len(districts)

    g = t.groupby(["Crop", "Crop_Variety", "Crop_Irrigation", "Crop_Season",
                   "Soil_Class", "Option", "Fertilizer"], dropna=False)["Quantity"]
    varies = int((g.nunique() > 1).sum())

    u = taluka_universe()
    no_texture = int(u["Soil_Texture"].isna().sum())

    return [
        f"**The fertiliser table is not complete across districts.** Only "
        f"{cells} of {total} (mapped crop x district) cells carry a published "
        f"recipe — {100 * cells / total:.1f}%. Linseed appears in 3 districts of 34, "
        f"sesame in 9. The plan does not mention this. Because doses genuinely "
        f"vary by district ({varies} of {len(g)} crop-context keys have more than "
        f"one distinct quantity across districts), borrowing a neighbour's recipe "
        f"silently would be wrong; the engine serves a state-wide median instead, "
        f"labelled as an estimate and carrying its own observed spread.",

        f"**Four talukas have no recorded soil texture** (Tumsar, Ausa, Latur, "
        f"Parseoni — all Alluvial). Imputed from the modal texture of their own "
        f"soil type and flagged with `texture_imputed`. ({no_texture} remain "
        f"missing after imputation.)",

        "**The plan's §6.6 crop spellings do not match the table.** It writes "
        "`Pearl millet`, `Pigeon pea`, `Finger millet`, `Tetraploid cotton`; the "
        "table uses title case. The ontology resolves case-insensitively and is "
        "unit-tested against the real 78-crop list.",

        "**The plan's §4.2 texture vocabulary does not match the data.** It "
        "assumes a Sandy -> Clayey scale; the data carries only Clayey, Loamy and "
        "two `-skeletal` variants. Skeletal soils hold >35% coarse fragments, so "
        "their fine-earth AWC is discounted ~40%.",

        "**Rabi cannot be scored on in-season rainfall.** Rabi rain averages "
        "~60 mm state-wide. Maharashtra's Rabi crops run on monsoon moisture "
        "stored in the profile, so the suitability gate adds the root-zone store "
        "to the season's rain. Without this the gate vetoes every Rabi crop in "
        "the state.",
    ]


def panel_summary() -> pd.DataFrame:
    """The eight-year panel, per year."""
    from src.data.admin_changes import audit
    return audit(load_apy_panel())


def source_inventory() -> pd.DataFrame:
    """Every delivered file actually used, and what it contributes."""
    rows = []
    for y, df in sorted(load_shc_cycles().items()):
        rows.append({"dataset": "Soil Health Card", "period": y,
                     "rows": len(df), "unit": "talukas"})
    for y, df in sorted(load_weather_years().items()):
        rows.append({"dataset": "Daily weather", "period": y,
                     "rows": len(df), "unit": "taluka-days"})
    apy = load_apy_panel()
    for y, g in apy.groupby("Year"):
        rows.append({"dataset": "Crop statistics (APY)", "period": y,
                     "rows": len(g), "unit": "district-crop-season"})
    return pd.DataFrame(rows)


def main() -> None:
    cov = coverage_report()
    out = [
        "# Data quality report",
        "",
        "Generated by `python -m src.data.quality_report`. Every structural claim in "
        "the project plan re-checked against the raw files.",
        "",
        "## The eight-year panel",
        "",
        "The original plan was written for a single year of crop statistics and "
        "named multi-year APY as the highest-value addition available. That data "
        "has arrived: **272 district-year label units instead of 34**. Full "
        "analysis in `reports/multiyear_upgrade.md`.",
        "",
        _md(panel_summary()),
        "",
        "### Every delivered file in use",
        "",
        "One delivered weather file is **excluded**: "
        "`maharashtra_daily_weather_taluka_2022-04-01_to_2023-03-31.csv` is a "
        "byte-identical duplicate of the 2025-26 file and its own Date column "
        "reads 2025-04-01..2026-03-31. Using it would double-count a year and "
        "manufacture a false claim of weather contemporaneous with the 2022-23 "
        "crop labels. `tests/test_panel.py` now asserts every weather file's "
        "internal dates match its filename.",
        "",
        _md(source_inventory()),
        "",
        "## Claims in the plan, verified against the data",
        "",
        _md(verified_claims()),
        "",
        "## The seven urban talukas",
        "",
        "They appear in the soil-type and weather files but carry no Soil Health "
        "Card record. Dropped with an explicit flag rather than imputed.",
        "",
        _md(urban_talukas()),
        "",
        "## The six duplicated taluka names",
        "",
        "Joining on `Taluka` alone silently corrupts these rows. Every join in this "
        "repository uses `(District, Taluka)`; `tests/test_data_contracts.py` enforces it.",
        "",
        _md(duplicate_talukas()),
        "",
        "## Fertiliser `Data_Flag` breakdown",
        "",
        "Filtered on `Data_Flag.isna()` before any per-hectare arithmetic. The "
        "per-tree rows cannot be converted to a per-hectare dose and the flagged "
        "rows reach 3,990 kg/ha.",
        "",
        _md(flag_breakdown()),
        "",
        "## The crop ontology",
        "",
        f"A raw string join matches only 8 of 25 APY crops. This mapping recovers "
        f"{cov['mapped_crops']}, covering **{cov['area_coverage_pct']}%** of "
        f"Maharashtra's cropped area.",
        "",
        _md(ontology_table()),
        "",
        f"Unmapped ({100 - cov['area_coverage_pct']:.1f}% of area, served from the "
        f"rule scorer only): " + ", ".join(f"`{c}`" for c in cov["unmapped_crops"]) + ".",
        "",
        "## Fertiliser recipe coverage by crop",
        "",
        _md(fertiliser_coverage()),
        "",
        "## Additional findings not stated in the plan",
        "",
    ]
    out += [f"{i}. {f}\n" for i, f in enumerate(additional_findings(), 1)]

    path = config.REPORTS / "data_quality.md"
    path.write_text("\n".join(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
