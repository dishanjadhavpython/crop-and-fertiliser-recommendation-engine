"""Controlled experiments for engine improvements.

    python -m src.eval.experiments

Every arm runs under the identical protocol the rest of the project uses —
GroupKFold by district, seed-bagged — so the deltas are comparable. Nothing
here is adopted unless it wins on this table.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src import config
from src.data.training_set import build, district_features, model_features
from src.eval import baselines
from src.eval.metrics import ranking_report
from src.features.agronomic_fit import FIT_COLUMNS, compute
from src.models import ranker

warnings.filterwarnings("ignore")


def with_fit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the crop-conditional agronomic fit block (early fusion)."""
    fit = compute(df, district_features())
    return pd.concat([df, fit], axis=1)


def arms(df: pd.DataFrame, seeds: int) -> pd.DataFrame:
    base = model_features(df)
    dff = with_fit_features(df)
    fit_cols = [c for c in FIT_COLUMNS if c in dff.columns]

    rows = []

    def run(name, frame, feats, **kw):
        s = ranker.cross_val_scores(frame, feats, seeds=seeds, **kw)
        frame = frame.assign(_s=s)
        rep = ranking_report(frame, "_s")
        rows.append({"Arm": name, "n feat": len(feats), **rep})
        print(f"  {name:<52} ndcg@5={rep['ndcg@5']:.3f}")
        return rep

    print("running arms ...")
    pop = ranking_report(df.assign(_s=baselines.popularity_prior_oof(df)), "_s")
    rows.append({"Arm": "Popularity prior (the bar)", "n feat": 0, **pop})
    print(f"  {'Popularity prior (the bar)':<52} ndcg@5={pop['ndcg@5']:.3f}")

    run("A. Current ranker (district features only)", df, base)
    run("B. + agronomic fit features  [early fusion]", dff, base + fit_cols)
    run("C. Agronomic fit features ONLY", dff, ["crop_id", "season_id"] + fit_cols)
    run("D. + SHC sample weighting", dff, base + fit_cols, use_weights=True)
    run("E. + monotone constraint on fit_min", dff, base + fit_cols,
        monotone=True)

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    df = build()
    table = arms(df, args.seeds)
    out = config.REPORTS / "experiments.md"
    out.write_text("# Engine improvement experiments\n\n"
                   "GroupKFold by district, "
                   f"{args.seeds}-seed bagging, identical to every other table.\n\n"
                   + table.to_markdown(index=False, floatfmt=".3f") + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
