"""The model tournament — LightGBM vs CatBoost vs TabPFN v2, on identical folds.

    python -m src.eval.model_selection                 # every stage
    python -m src.eval.model_selection --stage yield   # one of them
    python -m src.eval.model_selection --budget-min 5  # abandon a slow arm sooner

Writes ``reports/model_selection.md`` and ``artifacts/model_selection.json``;
``pipeline.train()`` reads the JSON to decide what to fit, and the cache key
hashes it, so a change of model can never be served from a stale artifact.

The rules exist because 34 districts is a small sample and it is easy to talk
yourself into a winner:

* **Identical everything.** Same folds, same features, same seeds per arm.
* **Significance over districts, not queries.** A thousand ranking queries are
  not a thousand independent facts; every query from one district shares its
  soil, climate and label history. Arms are compared with a paired test over
  the 34 district means, Holm-corrected across arms.
* **Ties go to the incumbent.** A challenger replaces LightGBM only if it wins
  significantly, breaks no hard gate, and stays inside the serving latency
  budget. Otherwise the simpler, faster model already in place stays.
* **Speed is reported beside skill.** A model that cannot answer a farmer in a
  second is not a candidate for this product whatever it scores, so fit time
  and prediction latency sit in the same table as NDCG.
* **An arm that cannot run says so.** Time budget exceeded, context limit,
  weights unavailable — the reason is recorded and published, never silently
  dropped.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src import config
from src.eval.metrics import district_paired_test, holm, per_query_ndcg, ranking_report
from src.eval.splits import forward_chaining, group_kfold
from src.models import backends as B

INCUMBENT = "lightgbm"
#: a recommendation must come back inside this, so a model that cannot is out
LATENCY_BUDGET_S = 1.0

#: The yield intervals are nominally 80%, and `coverage_80` is measured against
#: this. It is here rather than inline because the adoption rule now reads it.
COVERAGE_TARGET = 0.80
#: How much further from that target a challenger's coverage may sit before its
#: win on the point estimate stops counting. Small on purpose: the farmer is
#: shown the interval, so an interval that overstates its confidence is a
#: different kind of wrong from a point estimate that is merely less sharp.
COVERAGE_TOLERANCE = 0.03


def _isolated(worker, payload):
    """Run one arm in its own interpreter, then throw that interpreter away.

    LightGBM ships libomp and CatBoost ships TBB. On macOS, once both thread
    pools have been used in a single process they deadlock against each other:
    the first attempt at this tournament sat for 52 minutes at 11% CPU with
    every worker — the main thread included — parked in `__kmp_suspend_64`, and
    had to be killed. Setting `OMP_NUM_THREADS` did not help; it deadlocked
    again at the same point. Spawning a fresh process per arm is what actually
    fixes it, because only one of the two runtimes is ever loaded.

    The cost is pickling the training frame into each arm, which is seconds
    against fits measured in minutes.
    """
    import multiprocessing as mp

    # The budget is enforced *here*, by the parent, not only inside the arm.
    # `_oof_ranking` checks its clock between folds, which is no help at all if
    # a child wedges before finishing the first one — and one did: the TabPFN
    # arm sat for five minutes at 0.94s of CPU, no sockets open and no weights
    # touched, deadlocked in the pool handoff after LightGBM and CatBoost had
    # both gone through the same path. A blocking `pool.apply` turned that into
    # a hung tournament rather than a failed arm.
    timeout = payload[-1] if isinstance(payload[-1], (int, float)) else None
    context = mp.get_context("spawn")
    pool = context.Pool(1)
    try:
        return pool.apply_async(worker, (payload,)).get(
            timeout=None if timeout is None else timeout + 60)
    except mp.TimeoutError as exc:
        pool.terminate()
        raise TimeoutError(
            f"the arm produced nothing within its {timeout/60:.0f} min budget and "
            f"was terminated; recorded rather than allowed to hang the tournament"
        ) from exc
    finally:
        # `terminate` rather than `close`: idempotent, and the arm may have been
        # killed from outside, in which case there is nothing left to close.
        pool.terminate()
        pool.join()


def _ranking_arm(payload):
    """One ranking arm end to end. Module level so `spawn` can import it."""
    df, feats, name, seeds, budget_s = payload
    factory = {
        "lightgbm": lambda: B.LGBMRankerBackend(seeds=seeds),
        "catboost": lambda: B.CatBoostRankerBackend(seeds=seeds),
        "tabpfn": lambda: B.TabPFNRankerBackend(),
    }[name]
    return _oof_ranking(df, feats, forward_chaining, factory, budget_s)


def _yield_arm(payload):
    """One quantile arm end to end. Module level so `spawn` can import it."""
    d, cold, name, budget_s = payload
    factory = {"lightgbm": B.LGBMQuantileBackend, "catboost": B.CatBoostQuantileBackend,
               "tabpfn": B.TabPFNQuantileBackend}[name]
    preds = {a: np.full(len(d), np.nan) for a in config.QUANTILES}
    fit_s, started, failed = 0.0, time.time(), None
    try:
        for tr, te in group_kfold(d):
            t0 = time.time()
            model = factory().fit(d.iloc[tr][cold], d.iloc[tr]["yield_z"])
            fit_s += time.time() - t0
            got = model.predict_quantiles(d.iloc[te][cold])
            for a in config.QUANTILES:
                preds[a][te] = got[a]
            if time.time() - started > budget_s:
                failed = (f"exceeded the {budget_s/60:.0f} min budget after "
                          f"{fit_s:.0f}s of fitting")
                break
    except Exception as exc:                            # noqa: BLE001 - record, never raise
        failed = f"{type(exc).__name__}: {str(exc)[:300]}"
    return preds, fit_s, failed


# ------------------------------------------------------------------ ranking --
def _oof_ranking(df: pd.DataFrame, feats: list[str], splitter, factory,
                 budget_s: float) -> tuple[np.ndarray, dict]:
    """Out-of-fold scores from one backend, abandoning the arm if it overruns."""
    from src.models import ranker

    scores = np.full(len(df), np.nan)
    fit_s = 0.0
    started = time.time()
    for tr, te in splitter(df):
        idx, groups = ranker._grouped(df, tr)
        t0 = time.time()
        model = factory().fit(df.iloc[idx][feats], df.iloc[idx]["relevance"], groups)
        fit_s += time.time() - t0
        scores[te] = model.predict(df.iloc[te][feats])
        if time.time() - started > budget_s:
            return scores, {"status": "abandoned",
                            "note": f"exceeded the {budget_s/60:.0f} min budget after "
                                    f"{fit_s:.0f}s of fitting; folds left unrun"}
    t0 = time.time()
    model.predict(df.iloc[:1][feats])
    latency = time.time() - t0
    return scores, {"status": "ok", "fit_s": round(fit_s, 1),
                    "latency_s": round(latency, 4)}


def ranking_tournament(seeds: int = 1, budget_s: float = 1200.0,
                       arms: tuple[str, ...] = ("lightgbm", "catboost", "tabpfn")) -> dict:
    """Each ranker under forward chaining — the protocol a deployment faces."""
    from src.data.training_set import build, model_features
    from src.eval import engine_eval
    from src.pipeline import attach_fit_features

    df = attach_fit_features(build()).reset_index(drop=True)
    feats = model_features(df)
    rules = engine_eval.rules_and_vetoes(df)
    tested = np.zeros(len(df), dtype=bool)
    for _tr, te in forward_chaining(df):
        tested[te] = True

    out = {"protocol": "forward chaining", "n_queries": int(
        df[tested].groupby(["District", "Season", "Year"]).ngroups), "arms": {}}
    frame = df.loc[tested, ["District", "Season", "Year", "Crop", "relevance"]].copy()

    for name in arms:
        try:
            print(f"  {name} ...", flush=True)
            scores, meta = _isolated(_ranking_arm, (df, feats, name, seeds, budget_s))
        except Exception as exc:                       # noqa: BLE001 - record, never raise
            out["arms"][name] = {"status": "failed",
                                 "note": f"{type(exc).__name__}: {str(exc)[:300]}"}
            continue
        col = f"score_{name}"
        frame[col] = scores[tested]
        if meta["status"] == "ok":
            meta["ndcg@5"] = ranking_report(frame, col)["ndcg@5"]
        out["arms"][name] = meta

    # head to head against the incumbent, over districts
    ranked = [n for n, m in out["arms"].items() if m.get("status") == "ok"]
    if INCUMBENT in ranked:
        raw = {}
        for name in ranked:
            if name == INCUMBENT:
                continue
            raw[name] = district_paired_test(frame, f"score_{INCUMBENT}", f"score_{name}")
        adjusted = holm({k: v["p_wilcoxon"] for k, v in raw.items()}) if raw else {}
        for name, test in raw.items():
            test["p_holm"] = adjusted[name]
            out["arms"][name]["vs_incumbent"] = test
    out["frame"] = frame
    return out


# -------------------------------------------------------------------- yield --
def pinball(y, pred, alpha: float) -> float:
    d = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def yield_tournament(budget_s: float = 1200.0,
                     arms: tuple[str, ...] = ("lightgbm", "catboost", "tabpfn")) -> dict:
    """Each quantile model on the cold-start regime — GroupKFold by district."""
    from src.data.training_set import build
    from src.eval.metrics import within_crop_spearman
    from src.models import yield_regimes
    from src.pipeline import attach_fit_features

    d = yield_regimes.prepare(attach_fit_features(build())).reset_index(drop=True)
    cold, _warm = yield_regimes.feature_sets(d)
    out = {"protocol": "GroupKFold by district (cold start)", "n_rows": len(d), "arms": {}}

    for name in arms:
        print(f"  {name} ...", flush=True)
        try:
            preds, fit_s, failed = _isolated(_yield_arm, (d, cold, name, budget_s))
        except Exception as exc:                        # noqa: BLE001 - record, never raise
            preds = {a: np.full(len(d), np.nan) for a in config.QUANTILES}
            fit_s, failed = 0.0, f"{type(exc).__name__}: {str(exc)[:300]}"

        if failed:
            out["arms"][name] = {"status": "abandoned" if "budget" in failed else "failed",
                                 "note": failed}
            continue

        got = d.assign(_p=preds[0.5]).dropna(subset=["_p"])
        out["arms"][name] = {
            "status": "ok",
            "fit_s": round(fit_s, 1),
            "within_crop_rho": within_crop_spearman(got, "_p"),
            "pinball_p50": round(pinball(got["yield_z"], got["_p"], 0.5), 4),
            "coverage_80": round(float(((d["yield_z"] >= preds[0.1])
                                        & (d["yield_z"] <= preds[0.9])).mean()), 3),
        }
    return out


# ------------------------------------------------------------------- report --
def _md(rows: list[dict]) -> str:
    return pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f")


def decide(ranking: dict, yielding: dict) -> dict:
    """Apply the adoption rule. A tie goes to the incumbent, always."""
    chosen = {"ranker": INCUMBENT, "yield": INCUMBENT, "why": {}}

    best, reason = INCUMBENT, "no challenger cleared the bar"
    for name, meta in ranking["arms"].items():
        test = meta.get("vs_incumbent")
        if not test:
            continue
        wins = test["delta"] > 0 and test["p_holm"] < 0.05 and test["ci_low"] > 0
        fast = meta.get("latency_s", 1e9) <= LATENCY_BUDGET_S
        if wins and fast:
            best = name
            reason = (f"+{test['delta']:.4f} NDCG@5 over {INCUMBENT}, "
                      f"Holm p={test['p_holm']:.3g}, CI excludes zero, "
                      f"latency {meta['latency_s']}s")
        elif wins and not fast:
            reason = f"{name} won but needs {meta.get('latency_s')}s per answer"
    chosen["ranker"], chosen["why"]["ranker"] = best, reason

    ok = {n: m for n, m in yielding["arms"].items() if m.get("status") == "ok"}
    if ok:
        winner = max(ok, key=lambda n: ok[n]["within_crop_rho"])
        incumbent = ok.get(INCUMBENT, {})
        margin = ok[winner]["within_crop_rho"] - incumbent.get("within_crop_rho", -9)

        # Coverage, not only rho. Ranking this stage on the point estimate alone
        # is how it first chose a model whose 80% interval covered 62% against
        # the incumbent's 71.5%: a sharper prediction bought with an interval
        # that misstates its own confidence. Served coverage error is a
        # scorecard sub-score in its own right, so the criterion that decides
        # this stage has to contain it.
        def miss(meta):
            return abs(meta.get("coverage_80", 0.0) - COVERAGE_TARGET)

        if winner == INCUMBENT:
            chosen["yield"] = winner
            chosen["why"]["yield"] = f"within-crop rho {ok[winner]['within_crop_rho']:.3f}"
        elif margin < 0.02:
            chosen["why"]["yield"] = (f"{winner} led by {margin:.3f} rho, inside the noise "
                                      f"at 34 districts; incumbent kept")
        elif miss(ok[winner]) > miss(incumbent) + COVERAGE_TOLERANCE:
            chosen["why"]["yield"] = (
                f"{winner} led by {margin:.3f} rho but its {COVERAGE_TARGET:.0%} interval "
                f"covers {ok[winner].get('coverage_80', float('nan')):.3f} against "
                f"{incumbent.get('coverage_80', float('nan')):.3f}; a sharper point "
                f"estimate is not worth an interval that overstates its confidence, "
                f"so the incumbent is kept")
        else:
            chosen["yield"] = winner
            chosen["why"]["yield"] = (
                f"within-crop rho {ok[winner]['within_crop_rho']:.3f}, "
                f"coverage {ok[winner].get('coverage_80', float('nan')):.3f}")
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["ranking", "yield", "both"], default="both")
    p.add_argument("--arms", default="lightgbm,catboost,tabpfn")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--budget-min", type=float, default=20.0,
                   help="per-arm wall clock before the arm is abandoned and recorded as such")
    args = p.parse_args()

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    budget = args.budget_min * 60
    print(f"backends available: {B.available()}", flush=True)

    ranking = {"arms": {}}
    yielding = {"arms": {}}
    if args.stage in ("ranking", "both"):
        print("ranking tournament (forward chaining) ...", flush=True)
        ranking = ranking_tournament(seeds=args.seeds, budget_s=budget, arms=arms)
    if args.stage in ("yield", "both"):
        print("yield tournament (GroupKFold, cold start) ...", flush=True)
        yielding = yield_tournament(budget_s=budget, arms=arms)

    chosen = decide(ranking, yielding)

    out = ["# Model selection",
           "",
           "LightGBM, CatBoost and TabPFN v2 on identical folds, features and seeds. "
           "Significance is a paired test over the 34 districts, Holm-corrected; a "
           "challenger replaces the incumbent only on a significant win inside the "
           f"{LATENCY_BUDGET_S:.0f}s serving budget, and a tie goes to the model already "
           "in place. TabPFN is pinned to its v2 weights, the only version whose licence "
           "permits commercial use.",
           ""]

    if ranking.get("arms"):
        out += ["## Ranking — forward chaining", ""]
        rows = []
        for name, m in ranking["arms"].items():
            test = m.get("vs_incumbent", {})
            rows.append({"Model": name, "Status": m.get("status", "?"),
                         "NDCG@5": m.get("ndcg@5", float("nan")),
                         "Fit s": m.get("fit_s", float("nan")),
                         "Latency s": m.get("latency_s", float("nan")),
                         "Δ vs incumbent": test.get("delta", float("nan")),
                         "Holm p": test.get("p_holm", float("nan")),
                         "Note": m.get("note", "")})
        out += [_md(rows), ""]

    if yielding.get("arms"):
        out += ["## Yield — cold start, GroupKFold by district", ""]
        rows = [{"Model": n, "Status": m.get("status", "?"),
                 "Within-crop rho": m.get("within_crop_rho", float("nan")),
                 "Pinball p50": m.get("pinball_p50", float("nan")),
                 "80% coverage": m.get("coverage_80", float("nan")),
                 "Fit s": m.get("fit_s", float("nan")), "Note": m.get("note", "")}
                for n, m in yielding["arms"].items()]
        out += [_md(rows), ""]

    out += ["## Chosen", "",
            f"- **ranker**: {chosen['ranker']} — {chosen['why'].get('ranker', '')}",
            f"- **yield**: {chosen['yield']} — {chosen['why'].get('yield', '')}", ""]

    (config.REPORTS / "model_selection.md").write_text("\n".join(out))
    payload = {"chosen": chosen,
               "ranking": {k: v for k, v in ranking.items() if k != "frame"},
               "yield": yielding}
    (config.ARTIFACTS / "model_selection.json").write_text(json.dumps(payload, indent=1,
                                                                     default=str))
    print(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
