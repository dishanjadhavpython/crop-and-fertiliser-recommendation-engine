"""Reproduce the §7 benchmark tables (plan §10).

    python -m src.eval.ablation

Runs the full evaluation suite and writes markdown tables to ``reports/``.
This is the section examiners read, so every number here is produced by the
same protocol the rest of the project uses: GroupKFold by district, no
exceptions.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import config
from src.data.training_set import build, model_features
from src.eval import baselines
from src.eval.metrics import (
    interval_coverage,
    per_crop_spearman,
    r2,
    ranking_report,
    within_crop_spearman,
)
from src.eval.splits import group_kfold, leave_one_district_out, spatial_block_folds
from src.features import agroclimate, interactions, soil_health, soil_physical
from src.models import blend, conformal, ranker, yield_quantile

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- blocks ----
def _block_columns() -> dict[str, list[str]]:
    """Feature names belonging to each engineering block, for the ablation."""
    a = [c for c in soil_health.build().columns if c not in config.KEY]
    b = [c for c in soil_physical.build().columns if c not in config.KEY]
    c = [c for c in agroclimate.build().columns if c not in config.KEY]
    az = [c for c in soil_health.build(with_context_z=True).columns
          if c.startswith(("z_state_", "z_district_"))]
    return {"A": a, "B": b, "C": c, "context_z": az}


def _cv_yield(df: pd.DataFrame, feats: list[str], target: str = "yield_z",
              n_splits: int = config.N_FOLDS, seeds: int = 3, **params) -> np.ndarray:
    """Out-of-fold predictions for a plain regressor, seed-bagged."""
    acc = np.zeros(len(df))
    for s in range(seeds):
        pred = np.full(len(df), np.nan)
        for tr, te in group_kfold(df, n_splits):
            m = LGBMRegressor(
                **{**yield_quantile.QUANTILE_PARAMS, "objective": "regression", **params},
                random_state=config.SEED + s,
            )
            m.fit(df.iloc[tr][feats], df.iloc[tr][target])
            pred[te] = m.predict(df.iloc[te][feats])
        acc += pred
    return acc / seeds


# ------------------------------------------------------------- §7.0 mirage --
def headline_mirage(y: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """The result that reshapes the whole design (plan §1)."""
    full = _cv_yield(y, feats, target="Yield")
    ident = _cv_yield(y, ["crop_id", "season_id"], target="Yield")
    honest = _cv_yield(y, feats, target="yield_z")
    return pd.DataFrame([
        {"Target": "Raw yield (t/ha), soil + weather features", "R2": r2(y["Yield"], full),
         "Verdict": "the mirage"},
        {"Target": "Raw yield (t/ha), crop + season identity ONLY", "R2": r2(y["Yield"], ident),
         "Verdict": "almost the same — the model learned crop scale, not agronomy"},
        {"Target": "Within-crop z-scored yield, soil + weather", "R2": r2(y["yield_z"], honest),
         "Verdict": "the honest signal"},
    ])


# --------------------------------------------------------- §7.1 ablation ----
def block_ablation(y: pd.DataFrame) -> pd.DataFrame:
    blocks = _block_columns()
    present = set(y.columns)

    def keep(cols):
        return [c for c in cols if c in present and pd.api.types.is_numeric_dtype(y[c])]

    ident = ["crop_id", "season_id"]
    a, b, c = keep(blocks["A"]), keep(blocks["B"]), keep(blocks["C"])
    # Latitude/Longitude belong to block D: they are the spatial coordinates
    # the interaction block reasons over. Excluding them here would make this
    # table's final row a different model from the one every other table uses.
    d = [x for x in model_features(y) if x not in set(a) | set(b) | set(c) | set(ident)]

    # Every step is expressed as an ordered subset of the canonical feature
    # order. LightGBM's column subsampling depends on column order, so the same
    # feature *set* in a different order scores up to 0.016 R2 differently —
    # the same magnitude as every architectural lever in §7.2. Fixing the order
    # is what makes this table comparable with the others.
    canonical = model_features(y)
    order = {f: i for i, f in enumerate(canonical)}

    def ordered(*groups):
        chosen = {f for g in groups for f in g if f in order}
        return sorted(chosen, key=order.__getitem__)

    steps = [
        ("Null model (predict the mean)", []),
        ("Crop + season identity only", ordered(ident)),
        ("+ Soil Health Card (12 components)", ordered(ident, a)),
        ("+ Soil physical", ordered(ident, a, b)),
        ("+ Agro-climatic (engineered)", ordered(ident, a, b, c)),
        ("+ Agronomic interactions", ordered(ident, a, b, c, d)),
    ]

    rows, prev = [], None
    for name, feats in steps:
        if not feats:
            pred = np.full(len(y), y["yield_z"].mean())
        else:
            pred = _cv_yield(y, feats)
        col = "_abl"
        y[col] = pred
        score = r2(y["yield_z"], pred)
        rows.append({
            "Feature set": name,
            "R2": score,
            "Delta": None if prev is None else round(score - prev, 3),
            "Per-crop rho": within_crop_spearman(y, col),
            "n feat": len(feats),
        })
        prev = score
    y.drop(columns=["_abl"], inplace=True)
    return pd.DataFrame(rows)


def spatial_and_context_ablation(y: pd.DataFrame, y_full: pd.DataFrame) -> pd.DataFrame:
    """The two blocks the plan predicts will *hurt* at n=34."""
    base = model_features(y)
    rows = [{"Feature set": "Best set (blocks A-D)", "R2": r2(y["yield_z"], _cv_yield(y, base)),
             "n feat": len(base)}]
    extra = model_features(y_full)
    knn = [c for c in extra if c.startswith(("knn5_", "anom_"))]
    ctx = [c for c in extra if c.startswith(("z_state_", "z_district_"))]
    for name, cols in (("+ Spatial kNN smoothing", knn), ("+ Contextual z-scores", ctx)):
        feats = base + [c for c in cols if c in y_full.columns]
        rows.append({"Feature set": name,
                     "R2": r2(y_full["yield_z"], _cv_yield(y_full, feats)),
                     "n feat": len(feats)})
    return pd.DataFrame(rows)


# ------------------------------------------------------ §7.2 model levers ---
def model_levers(y: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    rows = [{"Lever": f"Baseline, {len(feats)} features",
             "R2": r2(y["yield_z"], _cv_yield(y, feats))}]

    sel = yield_quantile.cross_val_quantiles(y, feats, seeds=1,
                                             select_top=config.N_FEATURES_SELECTED)
    rows.append({"Lever": f"Nested importance selection -> top {config.N_FEATURES_SELECTED}",
                 "R2": r2(y["yield_z"], sel["p50"])})

    for name, params in (
        ("Stronger regularisation (leaves=7, L2=5)", {"num_leaves": 7, "reg_lambda": 5.0}),
        ("Very strong regularisation (leaves=4)", {"num_leaves": 4, "reg_lambda": 5.0}),
        ("DART boosting", {"boosting_type": "dart"}),
    ):
        rows.append({"Lever": name, "R2": r2(y["yield_z"], _cv_yield(y, feats, **params))})

    for n in (1, config.N_SEEDS):
        rows.append({"Lever": f"Seed bagging, {n} seed(s)",
                     "R2": r2(y["yield_z"], _cv_yield(y, feats, seeds=n))})
    return pd.DataFrame(rows)


# ------------------------------------------------- §7.3 ranking benchmark ---
def ranking_benchmark(df: pd.DataFrame, feats: list[str], seeds: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["score_random"] = baselines.random_baseline(df)
    df["score_popularity"] = baselines.popularity_prior_oof(df)
    df["score_rules"] = baselines.rule_score_baseline(df)

    yz = _cv_yield(df[df["has_yield"]].reset_index(drop=True), feats)
    df["score_yield"] = np.nan
    df.loc[df["has_yield"], "score_yield"] = yz
    df["score_yield"] = df["score_yield"].fillna(df["score_yield"].min() - 1)

    df["score_area_reg"] = _cv_yield(df, feats, target="area_share")
    df["score_ident"] = ranker.cross_val_scores(df, ["crop_id", "season_id"], seeds=seeds)
    df["score_ranker"] = ranker.cross_val_scores(df, feats, seeds=seeds)

    # The engine: early fusion + per-crop blend weights from RANKING skill.
    # The weights are CROSS-FITTED — each fold's weights are learned only from
    # the other folds' districts. The first version fitted them on the same
    # out-of-fold scores it then evaluated, which flattered the blend.
    from src.eval import engine_eval
    from src.pipeline import attach_fit_features
    fused = attach_fit_features(df.drop(columns=[c for c in df.columns
                                                 if c.startswith("fit_")], errors="ignore"))
    ffeats = model_features(fused)
    rules = engine_eval.rules_and_vetoes(fused)
    plain = engine_eval.evaluate(fused, ffeats, "gkf", seeds=seeds, rules=rules, served=False)
    df["score_engine"] = plain.frame["score_engine"].reindex(df.index).to_numpy()
    # ...and the list a farmer is actually shown: vetoed crops and the crops
    # the product cannot recommend (APY aggregates, tobacco) sink to the
    # bottom. Not comparable with the unsunk rows above it — scorecard.py
    # applies the same sink to every baseline for a like-for-like comparison.
    served = engine_eval.evaluate(fused, ffeats, "gkf", seeds=seeds, rules=rules, served=True)
    df["score_engine_served"] = served.frame["score_engine"].reindex(df.index).to_numpy()

    labels = [
        ("Random ranking", "score_random"),
        ("Yield-regression framing", "score_yield"),
        ("Area-share regression framing", "score_area_reg"),
        ("S2 agronomic rules only (no labels)", "score_rules"),
        ("LambdaMART, crop identity only", "score_ident"),
        ("Popularity prior, out-of-fold  <- the bar", "score_popularity"),
        ("LambdaMART + engineered features", "score_ranker"),
        ("Full engine (early fusion + cross-fitted blend)", "score_engine"),
        ("Engine as served (+ veto, unrecommendable crops sunk)", "score_engine_served"),
    ]
    rows = []
    for name, col in labels:
        rows.append({"Method": name, **ranking_report(df, col)})
    return pd.DataFrame(rows), df


# --------------------------------------------------- §7.4 per-crop skill ----
def per_crop_table(y: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    y = y.copy()
    y["pred"] = _cv_yield(y, feats)
    return per_crop_spearman(y, "pred")


# ---------------------------------------------- §7.5 protocol comparison ----
def protocol_comparison(y: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Why GroupKFold is not optional."""
    from sklearn.model_selection import KFold

    rows = []
    naive = np.full(len(y), np.nan)
    for tr, te in KFold(config.N_FOLDS, shuffle=True, random_state=config.SEED).split(y):
        m = LGBMRegressor(**{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
                          random_state=config.SEED)
        m.fit(y.iloc[tr][feats], y.iloc[tr]["yield_z"])
        naive[te] = m.predict(y.iloc[te][feats])
    rows.append({"Protocol": "Random KFold (LEAKS — districts split across folds)",
                 "R2": r2(y["yield_z"], naive)})
    rows.append({"Protocol": f"GroupKFold by district ({config.N_FOLDS} folds)",
                 "R2": r2(y["yield_z"], _cv_yield(y, feats))})

    for name, splitter in (("Leave-one-district-out (34 folds)", leave_one_district_out),
                           ("Spatial block CV (KMeans on centroids)", spatial_block_folds)):
        pred = np.full(len(y), np.nan)
        for tr, te in splitter(y):
            m = LGBMRegressor(**{**yield_quantile.QUANTILE_PARAMS, "objective": "regression"},
                              random_state=config.SEED)
            m.fit(y.iloc[tr][feats], y.iloc[tr]["yield_z"])
            pred[te] = m.predict(y.iloc[te][feats])
        rows.append({"Protocol": name, "R2": r2(y["yield_z"], pred)})
    return pd.DataFrame(rows)


# ------------------------------------------------------- §5.4 calibration ---
def calibration_table(y: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = y.copy()
    q = yield_quantile.cross_val_quantiles(y, feats, seeds=3)
    y = y.join(q)

    rows = [{"Interval": "Raw quantile model, nominal 80%",
             **interval_coverage(y["yield_z"], y["p10"], y["p90"])}]
    for alpha in config.CONFORMAL_ALPHAS:
        c = conformal.cross_val_conformal(y, "p10", "p90", alpha=alpha)
        y = y.join(c)
        lo, hi = c.columns
        rows.append({"Interval": f"Split-conformal, nominal {int((1-alpha)*100)}%",
                     **interval_coverage(y["yield_z"], y[lo], y[hi])})
    return pd.DataFrame(rows), y


# --------------------------------------------------- §6.5 cost optimiser ---
def cost_optimiser_benchmark(n: int = 400) -> pd.DataFrame:
    """Does the least-cost LP actually save money? (plan §6.5)

    An honest negative result: with only four straight fertilisers and three
    nutrient constraints the published Option 1 is already cost-optimal at
    standard subsidised prices. The LP earns its keep only under constraints
    the table cannot express.
    """
    from src.rules import fertiliser as fzz
    from src.rules.cost_optimiser import compare_with_table, optimise

    t = fzz._table()
    keys = t[["District", "Crop", "Crop_Variety", "Crop_Irrigation",
              "Crop_Season", "Soil_Class"]].drop_duplicates()
    rng = np.random.default_rng(config.SEED)
    sample = keys.iloc[rng.choice(len(keys), min(n, len(keys)), replace=False)]

    rows = []
    for r in sample.itertuples(index=False):
        opts, target = {}, None
        for opt in (1, 2):
            hit = fzz.lookup(
                r.District, r.Crop, r.Soil_Class, variety=r.Crop_Variety,
                irrigation=None if pd.isna(r.Crop_Irrigation) else r.Crop_Irrigation,
                season=None if pd.isna(r.Crop_Season) else r.Crop_Season, option=opt)
            if hit:
                opts[opt] = hit["products"]
                target = hit["target"]
        if not opts or target is None or target.get("N") is None:
            continue
        base = compare_with_table(opts, target)
        no_dap = optimise(target, available=["Urea", "SSP", "MOP"])
        with_s = optimise(target, require_sulphur_kg=20.0)
        rows.append({
            "table": base["cheapest_table_cost"],
            "saving": base["saving_inr_per_ha"],
            "no_dap": no_dap["cost_inr_per_ha"] if no_dap["feasible"] else np.nan,
            "with_s": with_s["cost_inr_per_ha"] if with_s["feasible"] else np.nan,
        })

    d = pd.DataFrame(rows)
    return pd.DataFrame([
        {"Scenario": "LP vs cheapest published option",
         "Mean cost delta vs table (INR/ha)": round(-d["saving"].mean(), 2),
         "Max saving found (INR/ha)": round(d["saving"].max(), 2),
         "LP solved": int(d["saving"].notna().sum()), "n": len(d)},
        {"Scenario": "DAP locally unavailable — table has no answer",
         "Mean cost delta vs table (INR/ha)": round((d["no_dap"] - d["table"]).mean(), 2),
         "Max saving found (INR/ha)": None,
         "LP solved": int(d["no_dap"].notna().sum()), "n": len(d)},
        {"Scenario": "20 kg S/ha required — table has no answer",
         "Mean cost delta vs table (INR/ha)": round((d["with_s"] - d["table"]).mean(), 2),
         "Max saving found (INR/ha)": None,
         "LP solved": int(d["with_s"].notna().sum()), "n": len(d)},
    ])


# ------------------------------------------------------------------ main ----
def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=3, help="seed-bagging repetitions")
    p.add_argument("--quick", action="store_true", help="skip the slowest tables")
    args = p.parse_args()

    print("building training frame ...")
    df = build()
    df_full = build(with_context_z=True, with_spatial=True)
    feats = model_features(df)
    y = yield_quantile.training_rows(df)
    y_full = yield_quantile.training_rows(df_full)
    yfeats = model_features(y)

    out = ["# Benchmark results",
           "",
           "Reproduces the tables in §7 of the project plan, on this repository's own "
           "feature store and evaluation harness.",
           "",
           f"Protocol: GroupKFold by district ({config.N_FOLDS} folds), "
           f"{args.seeds}-seed bagging, {y['District'].nunique()} districts x "
           f"{y['Year'].nunique()} years = {y.groupby(['District','Year']).ngroups} "
           f"district-year label units, {len(y)} rows with a recorded yield, "
           f"{len(df)} rows in the ranking candidate grid, "
           f"{df.groupby(['District','Season','Year']).ngroups} ranking queries.",
           ""]

    print("1/9 headline mirage ...")
    out += ["## The headline number is a mirage", "", _md(headline_mirage(y, yfeats)), ""]

    print("2/9 protocol comparison ...")
    out += ["## Why GroupKFold by district is not optional", "",
            _md(protocol_comparison(y, yfeats)), "",
            "With eight years of labels two further protocols become possible, "
            "and they are the ones that matter for deployment. **Forward "
            "chaining** trains on the past and ranks the next season — the "
            "question a farmer actually asks. **Grouped + temporal** shares "
            "neither a district nor a year between train and test, so neither "
            "spatial autocorrelation nor an era effect can carry a result. "
            "Both are reported in `reports/multiyear_upgrade.md`.", ""]

    print("3/9 feature block ablation ...")
    out += ["## 7.1 Feature block ablation", "", _md(block_ablation(y)), ""]

    print("4/9 spatial / context ablation ...")
    out += ["### Blocks the n=34 constraint rejects", "",
            _md(spatial_and_context_ablation(y, y_full)), ""]

    if not args.quick:
        print("5/9 model levers ...")
        out += ["## 7.2 Model-selection levers", "", _md(model_levers(y, yfeats)), ""]

    print("6/9 ranking benchmark ...")
    rank_tbl, scored = ranking_benchmark(df, feats, seeds=args.seeds)
    out += ["## 7.3 Ranking benchmark  <- the headline result", "", _md(rank_tbl), ""]

    print("7/9 per-crop skill ...")
    pc = per_crop_table(y, yfeats)
    out += ["## 7.4 Where the model works and where it fails", "", _md(pc), ""]

    alphas = blend.alpha_from_skill(pc)
    out += ["### Resulting per-crop blend weights (S5)", "",
            _md(blend.alpha_report(alphas, pc)), ""]

    print("8/9 calibration ...")
    cal, _ = calibration_table(y, yfeats)
    out += ["## 5.4 Conformal calibration", "", _md(cal), ""]

    print("9/9 cost optimiser ...")
    out += ["## 6.5 Does the least-cost LP save money?", "",
            "An honest negative result. With only four straight fertilisers "
            "(Urea, DAP, SSP, MOP) and three nutrient constraints, the government "
            "table's Option 1 is already cost-optimal at standard subsidised "
            "prices — the LP recovers it to within a rupee and never beats it. "
            "The optimiser earns its place only where the table cannot help: when "
            "a product is locally unavailable, or when the micronutrient layer "
            "requires sulphur that no published option supplies.", "",
            _md(cost_optimiser_benchmark()), ""]

    path = config.REPORTS / "benchmark_results.md"
    path.write_text("\n".join(out))
    scored.to_parquet(config.ARTIFACTS / "scored_ranking.parquet", index=False)
    pc.to_csv(config.ARTIFACTS / "per_crop_skill.csv", index=False)
    (config.ARTIFACTS / "blend_alphas.json").write_text(json.dumps(alphas, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
