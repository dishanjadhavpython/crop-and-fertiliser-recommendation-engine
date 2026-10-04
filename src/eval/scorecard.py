"""The engine's scorecard — what "9.5" means, measured the same way every time.

    python -m src.eval.scorecard --label "Phase 0 baseline"
    python -m src.eval.scorecard --quick --skip-tests --label "interim"

Every run appends a row to ``reports/scorecard_history.csv`` and rewrites
``reports/scorecard.md``, so progress across phases is one table.

Rules this file enforces
------------------------
* **Frozen anchors.** ``ANCHORS`` is hashed into
  ``reports/scorecard_anchors.sha256`` on the first run. A later run against
  different anchors refuses to score (``--rebase-anchors`` exists, and is
  recorded in the history, so moving the goalposts can never be silent).
* **The served code path.** Ranking goes through ``blend.served_scores`` via
  ``engine_eval.evaluate``; the metamorphic, input and explanation checks call
  ``pipeline.recommend`` itself.
* **Districts, not queries.** Significance is tested over the 34 districts.
* **Nothing unmeasured is scored.** A sub-score that cannot be measured yet
  (the soil photo, before the owner's dataset arrives) is printed as "not
  measured" and left out of the composite, with the measured share of the
  weight printed beside it.
* **Hard gates cap the composite at 6.0** if any one of them fails.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

#: Frozen before any improvement work (plan, "The scorecard"). Each sub-score
#: maps its measured value piecewise-linearly: ``zero`` -> 0, ``ten`` -> 10.
ANCHORS: dict[str, dict] = {
    "fc_margin": {"label": "Forward-chaining NDCG@5 margin over popularity",
                  "weight": 1.5, "zero": 0.05, "ten": 0.16},
    "gt_margin": {"label": "Grouped + temporal NDCG@5 margin over popularity",
                  "weight": 1.5, "zero": 0.0, "ten": 0.09},
    "false_veto": {"label": "False vetoes on >5% of district area",
                   "weight": 1.0, "zero": 0.03, "ten": 0.005},
    "metamorphic": {"label": "Metamorphic relations (M1-M3, M5, M6) pass rate",
                    "weight": 1.5, "zero": 0.90, "ten": 1.0},
    "fert_context": {"label": "Fertiliser season/irrigation context-match (M4)",
                     "weight": 1.0, "zero": 0.80, "ten": 1.0},
    "cold_rho": {"label": "Cold-start within-crop yield rho",
                 "weight": 0.75, "zero": 0.30, "ten": 0.55},
    "coverage_error": {"label": "Served-regime 90% interval coverage error",
                       "weight": 0.75, "zero": 0.10, "ten": 0.02},
    "soil_f1": {"label": "Soil photo macro-F1 on the owner's dataset (grouped CV)",
                "weight": 0.5, "zero": 0.60, "ten": 0.85},
    "soil_calibration": {"label": "Soil photo calibration (ECE) + non-soil rejection",
                         "weight": 0.5, "zero": 0.0, "ten": 1.0,
                         "ece_zero": 0.15, "ece_ten": 0.05,
                         "reject_zero": 0.0, "reject_ten": 0.95},
    "inputs_effective": {"label": "Inputs that measurably change the answer (of 15)",
                         "weight": 0.5, "zero": 0.0, "ten": 1.0},
    "explanation": {"label": "Explanations, provenance and Maharashtra crop count",
                    "weight": 0.5, "zero": 0.0, "ten": 1.0,
                    "crop_floor": 19, "crop_target": 45},
}
GATE_CAP = 6.0

ANCHOR_FILE = config.REPORTS / "scorecard_anchors.sha256"
HISTORY = config.REPORTS / "scorecard_history.csv"
REPORT = config.REPORTS / "scorecard.md"


def anchor_hash() -> str:
    blob = json.dumps({"anchors": ANCHORS, "cap": GATE_CAP}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def check_anchors(rebase: bool) -> str:
    h = anchor_hash()
    if ANCHOR_FILE.exists():
        frozen = ANCHOR_FILE.read_text().strip()
        if frozen != h:
            if not rebase:
                raise SystemExit(
                    "The scorecard anchors differ from the frozen set "
                    f"({frozen[:12]} vs {h[:12]}). Scores would not be comparable "
                    "with earlier rows. Re-run with --rebase-anchors only if the "
                    "change is deliberate; it will be recorded in the history.")
            ANCHOR_FILE.write_text(h)
            return "rebased"
        return "frozen"
    ANCHOR_FILE.write_text(h)
    return "frozen now"


def linear(value, zero: float, ten: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(min(10.0, max(0.0, 10.0 * (value - zero) / (ten - zero))))


# ------------------------------------------------------------ measurements --
def ranking_block(seeds: int):
    from src.data.training_set import build, model_features
    from src.eval import engine_eval
    from src.eval.metrics import ranking_report
    from src.pipeline import attach_fit_features

    df = attach_fit_features(build())
    feats = model_features(df)
    rules = engine_eval.rules_and_vetoes(df)
    vals, results = {}, {}
    for p in ("fc", "gt"):
        r = engine_eval.evaluate(df, feats, p, seeds=seeds, rules=rules)
        results[p] = r
        eng = ranking_report(r.frame, "score_engine")["ndcg@5"]
        pop = ranking_report(r.frame, "score_popularity")["ndcg@5"]
        vals[f"{p}_engine"], vals[f"{p}_popularity"] = eng, pop
        vals[f"{p}_margin"] = round(eng - pop, 4)
        if p == "fc":
            vals["fc_persistence"] = ranking_report(r.frame, "score_persistence")["ndcg@5"]
    return vals, results


def veto_block():
    from src.eval.gate_validation import score_panel, veto_metrics
    m = veto_metrics(score_panel())
    return float(m["false_veto_rate_area_gt_0.05"]), m


def _served_q(pipe, regime: str) -> float:
    q = pipe.conformal_q or {}
    for key in ((regime, 0.1), f"{regime}_0.1"):
        if key in q:
            return float(q[key])
    return float(q.get(0.1, 0.0))


def yield_block(pipe, seeds: int) -> dict:
    """Cold-start skill, and whether the *served* intervals cover what they claim."""
    from lightgbm import LGBMRegressor

    from src.data.training_set import build
    from src.eval.metrics import within_crop_spearman
    from src.eval.splits import forward_chaining, group_kfold
    from src.models import yield_quantile, yield_regimes
    from src.pipeline import attach_fit_features

    d = yield_regimes.prepare(attach_fit_features(build())).reset_index(drop=True)
    cold, warm = yield_regimes.feature_sets(d)
    p = yield_regimes._oof(d, cold, group_kfold, seeds)
    cold_rho = within_crop_spearman(d.assign(_p=p).dropna(subset=["_p"]), "_p")

    def band(frame, feats, splitter):
        lo = np.full(len(frame), np.nan)
        hi = np.full(len(frame), np.nan)
        for tr, te in splitter(frame):
            preds = [LGBMRegressor(**yield_quantile.QUANTILE_PARAMS, alpha=a,
                                   random_state=config.SEED)
                     .fit(frame.iloc[tr][feats], frame.iloc[tr]["yield_z"])
                     .predict(frame.iloc[te][feats])
                     for a in config.QUANTILES]
            stack = np.vstack(preds)
            lo[te], hi[te] = stack.min(axis=0), stack.max(axis=0)
        return lo, hi

    cov = {}
    warm_rows = d[d["has_history"] == 1].reset_index(drop=True)
    for regime, frame, feats, splitter in (("cold", d, cold, group_kfold),
                                           ("warm", warm_rows, warm, forward_chaining)):
        lo, hi = band(frame, feats, splitter)
        q = _served_q(pipe, regime)
        ok = ~np.isnan(lo)
        y = frame["yield_z"].to_numpy()[ok]
        cov[regime] = round(float(((y >= lo[ok] - q) & (y <= hi[ok] + q)).mean()), 4)
    return {"cold_rho": cold_rho, "coverage_cold": cov["cold"], "coverage_warm": cov["warm"],
            "coverage_error": round(float(np.mean([abs(v - 0.9) for v in cov.values()])), 4)}


def metamorphic_block(pipe, n_districts):
    from src.eval import metamorphic as mm

    talukas = mm.sample_talukas(n_districts)
    checks, recs = mm.run(pipe, talukas)
    applicable = checks[checks["passed"].notna()].copy()
    applicable["passed"] = applicable["passed"].astype(bool)
    core = applicable[applicable["test"] != "M4"]
    m4 = applicable[applicable["test"] == "M4"]
    meta = float(core["passed"].mean()) if len(core) else None
    fert = float(m4["passed"].mean()) if len(m4) else None
    return meta, fert, checks, recs, talukas


def recommendable_crop_count() -> int:
    """Crops the engine can place in a farmer's list (not the not-assessable ones)."""
    from src import pipeline
    from src.ontology.crop_map import CROP_ONTOLOGY, to_fertiliser_crop

    cands = set(CROP_ONTOLOGY) | set(pipeline._apy_crops())
    n = sum(1 for c in cands if to_fertiliser_crop(c) is not None)
    extra = getattr(pipeline, "horticulture_candidates", None)
    return n + (len(extra()) if callable(extra) else 0)


def explanation_block(recs) -> tuple[float, dict]:
    a = ANCHORS["explanation"]
    crops = [c for r in recs for c in r.crops]
    n = recommendable_crop_count()
    items = {
        "every crop carries a reason": all(bool(c.reason) for c in crops),
        "every assessed crop carries its factor scores":
            all(bool(c.factors) for c in crops if c.suitability_class not in ("?", None)),
        "every crop carries its climate/soil requirement ranges":
            all(bool(getattr(c, "requirements", None)) for c in crops),
        "every card reading carries its own source":
            all(isinstance(r.context.get("soil_test_sources"), dict) for r in recs),
        f"recommendable Maharashtra crops ({n}; {a['crop_floor']} -> {a['crop_target']})":
            float(min(1.0, max(0.0, (n - a["crop_floor"]) / (a["crop_target"] - a["crop_floor"])))),
    }
    return float(np.mean([float(v) for v in items.values()])), items


def soil_block() -> tuple[float | None, float | None, str]:
    """Read the soil-photo evaluation, if the owner-data training has produced one."""
    for path in (config.ARTIFACTS / "soil_photo_eval.json",
                 config.ROOT.parent / "ML" / "models" / "soil_eval.json"):
        if path.exists():
            m = json.loads(path.read_text())
            a = ANCHORS["soil_calibration"]
            f1 = m.get("grouped_cv_macro_f1")
            ece, rej = m.get("ece"), m.get("non_soil_rejection")
            cal = None
            if ece is not None and rej is not None:
                parts = [linear(ece, a["ece_zero"], a["ece_ten"]),
                         linear(rej, a["reject_zero"], a["reject_ten"])]
                cal = float(np.mean(parts)) / 10.0
            return f1, cal, f"from {path.name}"
    return None, None, ("not measured — soil-photo training waits for the owner's "
                        "Maharashtra dataset (research and plan/ML_PLAN.md §6)")


# ------------------------------------------------------------------- gates --
def _same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(abs(float(a[k]) - float(b[k])) < 1e-6 for k in a)


def _col_eq(t: pd.DataFrame, col: str, v) -> pd.Series:
    return t[col].isna() if pd.isna(v) else t[col] == v


def gate_exactness(n: int = 1000) -> tuple[bool, str]:
    from src.rules import fertiliser as fz

    t = fz._table()
    cols = ["District", "Crop", "Crop_Variety", "Crop_Irrigation", "Crop_Season",
            "Soil_Class", "Option"]
    keys = t[cols].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(config.SEED)
    sample = keys.iloc[rng.choice(len(keys), size=min(n, len(keys)), replace=False)]
    bad = 0
    for row in sample.itertuples(index=False):
        got = fz.lookup(row.District, row.Crop, row.Soil_Class,
                        variety=None if pd.isna(row.Crop_Variety) else row.Crop_Variety,
                        irrigation=None if pd.isna(row.Crop_Irrigation) else row.Crop_Irrigation,
                        season=None if pd.isna(row.Crop_Season) else row.Crop_Season,
                        option=int(row.Option))
        m = np.logical_and.reduce([_col_eq(t, c, getattr(row, c)) for c in cols])
        expect = dict(zip(t.loc[m, "Fertilizer"], t.loc[m, "Quantity"].astype(float)))
        bad += got is None or not _same(got["products"], expect)
    return bad == 0, f"{len(sample) - bad}/{len(sample)} sampled keys reproduce the table exactly"


def gate_veto_subset() -> tuple[bool, str]:
    """Irrigating must never veto a crop the rainfed answer allows."""
    from src.pipeline import _store
    from src.rules.crop_requirements import covered_crops
    from src.rules.suitability import score

    bad = []
    crops = covered_crops()
    for row in _store().to_dict("records"):
        for season in config.APY_SEASONS:
            for crop in crops:
                if score(crop, season, row, irrigated=True).vetoed and \
                        not score(crop, season, row, irrigated=False).vetoed:
                    bad.append(f"{row['Taluka']}/{season}/{crop}")
    note = f"{len(bad)} (taluka, season, crop) cells are vetoed only when irrigated"
    return not bad, note + (f", e.g. {bad[0]}" if bad else "")


def gate_photo() -> tuple[bool, str]:
    from src import pipeline
    if "soil_photo" not in inspect.signature(pipeline.recommend).parameters:
        return True, "vacuous: the engine takes no soil photograph yet"
    from src.rules import soil_fusion
    return soil_fusion.property_check()


def gate_cache() -> tuple[bool, str]:
    from src import pipeline

    required = sorted({p.resolve() for p in config.RAW.iterdir()
                       if p.suffix in (".csv", ".json")}
                      | {p.resolve() for p in (config.ROOT / "src").rglob("*.py")})
    sel = config.ARTIFACTS / "model_selection.json"
    if sel.exists():
        required.append(sel.resolve())
    if not hasattr(pipeline, "cache_inputs"):
        return False, (f"pipeline.cache_inputs() does not exist: the cache key hashes 9 "
                       f"modules and a parquet serving never reads, not the "
                       f"{len(required)} inputs the models depend on")
    covered = {Path(p).resolve() for p in pipeline.cache_inputs()}
    missing = [p for p in required if p not in covered]
    return not missing, (f"{len(required) - len(missing)}/{len(required)} inputs hashed"
                         + (f"; missing e.g. {missing[0].name}" if missing else ""))


def _pytest(args: list[str]) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider"],
                       cwd=config.ROOT, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return r.returncode == 0, (lines[-1] if lines else f"exit {r.returncode}")


def gates(run_tests: bool) -> dict[str, tuple]:
    g = {}
    g["all tests green"] = _pytest(["tests"]) if run_tests else (None, "not run (--skip-tests)")
    g["fertiliser table reproduced exactly"] = gate_exactness()
    g["irrigation never adds a veto"] = gate_veto_subset()
    g["photo never removes a veto"] = gate_photo()
    g["no-leakage tests pass"] = _pytest(["tests/test_no_leakage.py"])
    g["cache key covers every input"] = gate_cache()
    return g


# ------------------------------------------------------------------ report --
def _fmt(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def write_report(label, values, scores, comp, gate_res, details, elapsed, anchors_state):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_pass = sum(1 for ok, _ in gate_res.values() if ok is True)
    row = {"timestamp": now, "label": label,
           "composite": round(comp["capped"], 2), "composite_uncapped": round(comp["raw"], 2),
           "measured_weight": round(comp["measured_share"], 3),
           "gates": f"{n_pass}/{len(gate_res)}", "anchors": anchors_state,
           **{k: (None if values[k] is None else round(float(values[k]), 4)) for k in ANCHORS}}
    hist = pd.concat([pd.read_csv(HISTORY), pd.DataFrame([row])], ignore_index=True) \
        if HISTORY.exists() else pd.DataFrame([row])
    hist.to_csv(HISTORY, index=False)

    cap_note = (f" — **capped at {GATE_CAP:.1f}** because a hard gate failed"
                if comp["capped"] < comp["raw"] - 1e-9 else "")
    out = [
        "# Engine scorecard",
        "",
        "What \"9.5\" means for this engine, measured the same way on every run. "
        "Each sub-score maps its measured value piecewise-linearly from a frozen "
        "floor (0) to a frozen target (10); the composite is their weighted mean. "
        "Any failing hard gate caps the composite at 6.0. Unmeasured sub-scores are "
        "shown, never silently scored. "
        f"Anchors sha256 `{anchor_hash()[:16]}` ({anchors_state}).",
        "",
        f"## Latest: {label} — {now}",
        "",
        f"**Composite {comp['capped']:.2f} / 10** (uncapped {comp['raw']:.2f}; "
        f"{comp['measured_share']:.0%} of the weight measured){cap_note}. "
        f"Run time {elapsed/60:.1f} min.",
        "",
        "### Hard gates",
        "",
        _md(pd.DataFrame([{"Gate": k, "Status": {True: "pass", False: "FAIL", None: "not run"}[ok],
                           "Detail": detail} for k, (ok, detail) in gate_res.items()])),
        "",
        "### Sub-scores",
        "",
        _md(pd.DataFrame([{"Sub-score": a["label"], "Measured": _fmt(values[k], 4),
                           "0 at": a["zero"], "10 at": a["ten"], "Weight": a["weight"],
                           "Score": _fmt(scores[k], 2)} for k, a in ANCHORS.items()])),
        "",
    ]
    out += details
    out += ["## History", "", _md(hist), ""]
    REPORT.write_text("\n".join(out))
    (config.ARTIFACTS / "scorecard_latest.json").write_text(
        json.dumps({"row": row, "values": values, "scores": scores,
                    "gates": {k: [v[0], v[1]] for k, v in gate_res.items()}},
                   indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--label", default="interim")
    p.add_argument("--quick", action="store_true",
                   help="1 seed and 10 districts instead of 3 seeds and all 34")
    p.add_argument("--skip-tests", action="store_true",
                   help="do not run the full test suite (the gate reads 'not run')")
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--rebase-anchors", action="store_true")
    args = p.parse_args()

    anchors_state = check_anchors(args.rebase_anchors)
    seeds = args.seeds or (1 if args.quick else 3)
    n_districts = 10 if args.quick else None
    t0 = time.time()

    from src.pipeline import load_pipeline
    print("loading the served pipeline ...", flush=True)
    pipe = load_pipeline()

    print("1/6 ranking under forward chaining and grouped + temporal ...", flush=True)
    rank_vals, rank_res = ranking_block(seeds)
    print("2/6 gate false vetoes ...", flush=True)
    false_veto, veto_detail = veto_block()
    print("3/6 metamorphic relations ...", flush=True)
    meta, fert, checks, recs, talukas = metamorphic_block(pipe, n_districts)
    print("4/6 input effects ...", flush=True)
    from src.eval import metamorphic as mm
    effects = mm.input_effects(pipe, talukas[:6])
    inputs = float(effects["effective"].fillna(False).astype(bool).sum()) / len(effects)
    print("5/6 yield skill and served coverage ...", flush=True)
    yv = yield_block(pipe, seeds)
    expl, expl_items = explanation_block(recs)
    soil_f1, soil_cal, soil_note = soil_block()
    print("6/6 hard gates ...", flush=True)
    gate_res = gates(run_tests=not args.skip_tests)

    values = {
        "fc_margin": rank_vals["fc_margin"], "gt_margin": rank_vals["gt_margin"],
        "false_veto": false_veto, "metamorphic": meta, "fert_context": fert,
        "cold_rho": yv["cold_rho"], "coverage_error": yv["coverage_error"],
        "soil_f1": soil_f1, "soil_calibration": soil_cal,
        "inputs_effective": inputs, "explanation": expl,
    }
    scores = {k: linear(v, ANCHORS[k]["zero"], ANCHORS[k]["ten"]) if v is not None else None
              for k, v in values.items()}
    measured = {k: s for k, s in scores.items() if s is not None}
    w = sum(ANCHORS[k]["weight"] for k in measured)
    raw = sum(ANCHORS[k]["weight"] * s for k, s in measured.items()) / w if w else 0.0
    failed = any(ok is False for ok, _ in gate_res.values())
    comp = {"raw": raw, "capped": min(raw, GATE_CAP) if failed else raw,
            "measured_share": w / sum(a["weight"] for a in ANCHORS.values())}

    # ---- detail sections ----------------------------------------------------
    details = ["### Ranking — the list as served", ""]
    for key, r in rank_res.items():
        details += [f"**{r.label}** — {r.frame.groupby(['District','Season','Year']).ngroups} "
                    f"queries, {r.frame['District'].nunique()} districts", "",
                    _md(r.report), "",
                    f"Engine vs popularity over districts: Δ {r.paired['delta']:+.4f} "
                    f"(95% CI {r.paired['ci_low']:+.4f} … {r.paired['ci_high']:+.4f}), "
                    f"Wilcoxon p = {r.paired['p_wilcoxon']:.2g}, "
                    f"{r.paired['n_districts']} districts.", ""]
    details += [f"District persistence under forward chaining (grow what the district grew "
                f"last year): NDCG@5 {rank_vals['fc_persistence']:.3f}, against the engine's "
                f"{rank_vals['fc_engine']:.3f}.", ""]

    applicable = checks[checks["passed"].notna()]
    summary = []
    for test, g in checks.groupby("test"):
        app = g[g["passed"].notna()]
        fails = app[app["passed"] == False]  # noqa: E712
        summary.append({"Relation": test, "Checks": len(app),
                        "Passed": int((app["passed"] == True).sum()),  # noqa: E712
                        "Example failure": (f"{fails.iloc[0]['taluka']} {fails.iloc[0]['season']} "
                                            f"{fails.iloc[0]['crop'] or ''}: {fails.iloc[0]['detail']}")
                        if len(fails) else ("n/a — " + g.iloc[0]["detail"] if app.empty else "")})
    details += ["### Metamorphic relations", "",
                f"{len(talukas)} talukas (one per district) × Kharif and Rabi; "
                f"{len(applicable)} applicable checks.", "", _md(pd.DataFrame(summary)), ""]
    details += ["### Inputs that change the answer", "", _md(effects), ""]
    details += ["### Explanation checklist", "",
                _md(pd.DataFrame([{"Item": k, "Value": _fmt(float(v), 2)}
                                  for k, v in expl_items.items()])), ""]
    details += ["### Yield", "",
                f"Cold-start within-crop rho {yv['cold_rho']:.3f}. Served 90% intervals "
                f"cover {yv['coverage_cold']:.3f} (cold, GroupKFold) and "
                f"{yv['coverage_warm']:.3f} (warm, forward chaining).", ""]
    details += ["### Gate", "",
                f"False-veto rate (>5% of district area): {false_veto:.4f}; "
                f"veto rate {veto_detail['veto_rate']:.3f} over {veto_detail['scored_cells']} cells.", ""]
    details += ["### Soil photo", "", soil_note, ""]

    elapsed = time.time() - t0
    write_report(args.label, values, scores, comp, gate_res, details, elapsed, anchors_state)
    print(f"\ncomposite {comp['capped']:.2f} (uncapped {comp['raw']:.2f}, "
          f"{comp['measured_share']:.0%} measured) -> {REPORT}")


if __name__ == "__main__":
    main()
