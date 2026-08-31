"""Validate the S2 agronomic gate against eight years of revealed practice.

The gate is knowledge-based: FAO EcoCrop and ICAR envelopes, combined by
Liebig's minimum, using **no labels at all**. That is its strength — it covers
crops with no yield record and it is fully explainable — but it also means
nothing in the system was checking whether it is *right*.

The eight-year APY panel makes it falsifiable. If the gate is sound then crops
it rates suitable should be planted more often, and over more area, than crops
it vetoes. If it vetoes something a district plants across half its land, the
gate is wrong and the envelope needs repair.

Two metrics carry the verdict:

``false_veto_rate``
    Share of vetoed (district, crop, season, year) cells that were in fact
    planted over more than a threshold share of district area. A veto is a
    hard removal from the recommendation list, so a false veto is the most
    damaging error the system can make. This is the number to drive down.

``veto_rate``
    Share of all scored cells the gate rejects. A veto should be a rare,
    near-certain statement that the land cannot support the crop. A gate that
    vetoes most of the candidate space is not a safety net — it has quietly
    become the ranker, without being evaluated as one.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.training_set import build, district_features
from src.ontology.crop_map import to_fertiliser_crop
from src.rules.suitability import score

#: area shares above which a vetoed crop counts as a false veto
FALSE_VETO_THRESHOLDS = (0.0, 0.01, 0.05)


def score_panel(df: pd.DataFrame | None = None, irrigated: bool = False) -> pd.DataFrame:
    """Score the gate on every (district, crop, season, year) cell in the panel."""
    df = build() if df is None else df
    feats = district_features().set_index("District")

    cache: dict[tuple, object] = {}
    rows = []
    for r in df.itertuples(index=False):
        fert = to_fertiliser_crop(r.Crop)
        if fert is None or r.District not in feats.index:
            continue
        key = (r.District, r.Season, fert)
        if key not in cache:
            cache[key] = score(fert, r.Season, feats.loc[r.District].to_dict(),
                               irrigated=irrigated)
        s = cache[key]
        if not s.scored:
            continue
        rows.append({
            "District": r.District, "Year": r.Year, "Season": r.Season,
            "Crop": r.Crop, "suitability_class": s.suitability_class,
            "gate_score": s.score, "limiting_factor": s.limiting_factor,
            "vetoed": s.vetoed, "planted": r.planted, "area_share": r.area_share,
            "requires_irrigation": s.requires_irrigation,
            "score_irrigated": s.score_irrigated,
        })
    return pd.DataFrame(rows)


def agreement_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Planted rate and mean area share by suitability class.

    A sound gate produces a monotone table: rarer planting and smaller area as
    the class worsens.
    """
    t = (scored.groupby("suitability_class")
         .agg(cells=("planted", "size"),
              planted_rate=("planted", "mean"),
              mean_area_share=("area_share", "mean"),
              mean_area_share_if_planted=("area_share",
                                          lambda s: s[s > 0].mean() if (s > 0).any() else np.nan))
         .reindex(["S1", "S2", "S3", "N"]))
    return t.reset_index()


def veto_metrics(scored: pd.DataFrame) -> dict:
    """The headline numbers, and the ones the CI test guards."""
    vetoed = scored[scored["vetoed"]]
    out = {
        "scored_cells": int(len(scored)),
        "vetoed_cells": int(len(vetoed)),
        "veto_rate": round(float(scored["vetoed"].mean()), 4),
    }
    for thr in FALSE_VETO_THRESHOLDS:
        label = "planted" if thr == 0 else f"area_gt_{thr:g}"
        hits = (vetoed["area_share"] > thr) if thr else (vetoed["planted"] == 1)
        out[f"false_veto_{label}"] = int(hits.sum())
        out[f"false_veto_rate_{label}"] = round(
            float(hits.mean()) if len(vetoed) else 0.0, 4)
    return out


def worst_false_vetoes(scored: pd.DataFrame, threshold: float = 0.05,
                       top: int = 15) -> pd.DataFrame:
    """Crop x limiting-factor pairs responsible for the damaging vetoes."""
    bad = scored[scored["vetoed"] & (scored["area_share"] > threshold)]
    if bad.empty:
        return pd.DataFrame(columns=["Crop", "limiting_factor", "n", "mean_area_share"])
    return (bad.groupby(["Crop", "limiting_factor"])
            .agg(n=("area_share", "size"), mean_area_share=("area_share", "mean"),
                 max_area_share=("area_share", "max"))
            .sort_values("n", ascending=False).head(top).reset_index())


def factor_blame(scored: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Which limiting factor causes the most false vetoes, and how precise it is."""
    vetoed = scored[scored["vetoed"]]
    rows = []
    for factor, g in vetoed.groupby("limiting_factor"):
        bad = (g["area_share"] > threshold)
        rows.append({
            "limiting_factor": factor,
            "vetoes": len(g),
            "false_vetoes": int(bad.sum()),
            "false_veto_rate": round(float(bad.mean()), 4),
            "mean_area_when_wrong": round(float(g.loc[bad, "area_share"].mean()), 3)
            if bad.any() else 0.0,
        })
    return pd.DataFrame(rows).sort_values("false_vetoes", ascending=False).reset_index(drop=True)


def monotonicity(scored: pd.DataFrame) -> dict:
    """Does mean area share fall as the suitability class worsens?

    Finding from the first run: it does not — the ordering is inverted. The
    veto/no-veto split is sound but the four-level grade is not, which is why
    the grade should not be used to rank.
    """
    t = agreement_table(scored).set_index("suitability_class")
    order = ["S1", "S2", "S3", "N"]
    shares = [t.loc[c, "mean_area_share"] for c in order if c in t.index]
    planted = [t.loc[c, "planted_rate"] for c in order if c in t.index]
    return {
        "area_share_monotone_decreasing": bool(all(
            a >= b - 1e-9 for a, b in zip(shares, shares[1:]))),
        "planted_rate_monotone_decreasing": bool(all(
            a >= b - 1e-9 for a, b in zip(planted, planted[1:]))),
        "suitable_vs_vetoed_planted_ratio": round(
            float(np.mean(planted[:-1]) / planted[-1]), 2) if planted[-1] else None,
    }


def irrigation_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Cells that pass only with irrigation — and whether farmers do plant them.

    A high planted rate here is the validation: these are places where the
    land is fine and the water has to be supplied, and farmers evidently do
    supply it.
    """
    rows = []
    for flag, g in scored.groupby("requires_irrigation"):
        rows.append({
            "requires_irrigation": bool(flag),
            "cells": len(g),
            "planted_rate": round(float(g["planted"].mean()), 3),
            "mean_area_share": round(float(g["area_share"].mean()), 3),
        })
    by_season = (scored[scored["requires_irrigation"]]
                 .groupby("Season")
                 .agg(cells=("planted", "size"), planted_rate=("planted", "mean"))
                 .round(3).reset_index())
    out = pd.DataFrame(rows)
    out.attrs["by_season"] = by_season
    return out


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def report(seeds_note: str = "") -> str:
    rain = score_panel(irrigated=False)
    m_rain = veto_metrics(rain)
    mono = monotonicity(rain)

    out = [
        "# S2 gate validation against revealed practice",
        "",
        "The agronomic gate uses **no labels**. This report checks it against "
        "eight years of what Maharashtra's farmers actually planted "
        f"({m_rain['scored_cells']:,} district x crop x season x year cells).",
        "",
        "## Does the gate agree with practice?",
        "",
        _md(agreement_table(rain)),
        "",
        f"A crop the gate calls suitable is **"
        f"{mono['suitable_vs_vetoed_planted_ratio']}x** more likely to be planted "
        f"than one it vetoes. That is the gate working.",
        "",
        f"- planted rate falls monotonically S1->N: "
        f"**{mono['planted_rate_monotone_decreasing']}**",
        f"- mean area share falls monotonically S1->N: "
        f"**{mono['area_share_monotone_decreasing']}**",
        "",
        "## Veto quality",
        "",
        "The veto is decided on the *most favourable water scenario*, so it "
        "means \"this land cannot support this crop\" rather than \"cannot "
        "without irrigation\". Crops that fail only on water are reported as "
        "needing irrigation instead of being removed.",
        "",
        _md(pd.DataFrame([{"gate": "hard-veto (water-relieved)", **m_rain}])),
        "",
        "## The new middle category: viable, but only with irrigation",
        "",
        _md(irrigation_table(rain)),
        "",
        "## Which factor is responsible (rainfed)",
        "",
        _md(factor_blame(rain)),
        "",
        "## Worst false vetoes — crops the gate rejects but farmers plant heavily",
        "",
        _md(worst_false_vetoes(rain)),
        "",
    ]
    if seeds_note:
        out += [seeds_note, ""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(config.REPORTS / "gate_validation.md"))
    args = p.parse_args()

    text = report()
    path = config.REPORTS / "gate_validation.md" if args.out is None else args.out
    with open(path, "w") as f:
        f.write(text)

    scored = score_panel()
    scored.to_parquet(config.ARTIFACTS / "gate_scored_panel.parquet", index=False)
    m = veto_metrics(scored)
    print(f"wrote {path}")
    print(f"  veto rate            {m['veto_rate']:.1%}")
    print(f"  false veto (>1% area) {m['false_veto_rate_area_gt_0.01']:.2%}")
    print(f"  false veto (>5% area) {m['false_veto_rate_area_gt_0.05']:.2%}")


if __name__ == "__main__":
    main()
