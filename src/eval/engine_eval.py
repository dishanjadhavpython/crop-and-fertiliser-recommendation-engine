"""Evaluate the engine the way it is served — under any protocol.

The ranking benchmark used to score a quantity close to, but not the same as,
the list a farmer sees:

* it blended the learned and rule scores but never applied the hard veto,
  while serving removes vetoed crops from the list outright;
* it let crops the product can never recommend (the APY aggregates such as
  "Other Kharif pulses", and tobacco) compete for the top five, though serving
  moves them to ``not_assessable``;
* it fitted the per-crop blend weights on the very out-of-fold scores it then
  evaluated, which flatters the blend.

Here every protocol goes through ``blend.served_scores`` — the function serving
itself uses — with vetoed and non-recommendable crops sunk to the bottom, and
the blend weights are **cross-fitted**: the weights applied to a test fold are
learned only from rows outside it (for forward chaining, only from earlier
years). The non-recommendable mask is applied to every baseline too, so the
question each method answers is the same: how well does it order the crops this
product can actually recommend?
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from src import config
from src.eval import baselines
from src.eval.metrics import district_paired_test, ranking_report
from src.eval.splits import forward_chaining, group_kfold, grouped_and_temporal
from src.models import blend, ranker

PROTOCOLS = {
    "gkf": ("GroupKFold by district", group_kfold),
    "fc": ("Forward chaining (train past -> rank next year)", forward_chaining),
    "gt": ("Grouped + temporal (unseen district AND year)", grouped_and_temporal),
}

#: how far below every servable candidate a non-recommendable crop sinks
SINK = 5.0


def rules_and_vetoes(df: pd.DataFrame, irrigated: bool = True) -> pd.DataFrame:
    """The S2 rule score, the hard veto and servability for every row.

    ``servable`` mirrors ``pipeline.recommend``: a crop with neither an
    envelope nor a fertiliser recipe is reported as not assessable and never
    competes for a slot.
    """
    from src.data.training_set import district_features
    from src.ontology.crop_map import to_fertiliser_crop
    from src.rules.suitability import score

    feats = district_features().set_index("District")
    cache: dict[tuple, tuple] = {}
    rule, vetoed, servable = [], [], []
    for d, s, c in zip(df["District"], df["Season"], df["Crop"]):
        key = (d, s, c)
        if key not in cache:
            fert = to_fertiliser_crop(c)
            if fert is None or d not in feats.index:
                cache[key] = (0.0, False, fert is not None)
            else:
                r = score(fert, s, feats.loc[d].to_dict(), irrigated=irrigated)
                cache[key] = ((float(r.score) if r.scored else 0.0),
                              bool(r.scored and r.vetoed),
                              bool(r.scored) or fert is not None)
        a, b, c_ = cache[key]
        rule.append(a)
        vetoed.append(b)
        servable.append(c_)
    return pd.DataFrame({"score_rules": rule, "vetoed": vetoed, "servable": servable},
                        index=df.index)


def _fit_predict(df: pd.DataFrame, feats: list[str], tr: np.ndarray, te: np.ndarray,
                 seeds: int) -> np.ndarray:
    idx, groups = ranker._grouped(df, tr)
    pred = np.zeros(len(te))
    # ranker.params(), so the model measured here is configured exactly like the
    # one that serves — monotone constraints included
    p = ranker.params(feats)
    for s in range(seeds):
        m = LGBMRanker(**{**p, "random_state": config.SEED + s})
        m.fit(df.iloc[idx][feats], df.iloc[idx]["relevance"], group=groups)
        pred += m.predict(df.iloc[te][feats])
    return pred / seeds


@dataclass
class ProtocolResult:
    protocol: str
    label: str
    frame: pd.DataFrame          # test rows only, with every score column
    report: pd.DataFrame         # NDCG@5 / NDCG@3 / P@3 / Recall@5 per method
    paired: dict                 # engine vs popularity, over districts


def _sunk(score: pd.Series, servable: pd.Series) -> pd.Series:
    return score.where(servable, score - SINK)


def evaluate(df: pd.DataFrame, feats: list[str], protocol: str = "fc",
             seeds: int = 3, rules: pd.DataFrame | None = None,
             served: bool = True) -> ProtocolResult:
    """Out-of-sample served scores for one protocol, beside its baselines.

    ``served=False`` keeps only the cross-fitted blend weights and skips the
    veto and the non-recommendable sink, so the historical benchmark table can
    be reproduced on its own terms.
    """
    label, splitter = PROTOCOLS[protocol]
    df = df.reset_index(drop=True)
    rules = rules_and_vetoes(df) if rules is None else rules.reset_index(drop=True)
    folds = list(splitter(df))

    learned = pd.Series(np.nan, index=df.index)
    fold_of = pd.Series(-1, index=df.index)
    for k, (tr, te) in enumerate(folds):
        learned.iloc[te] = _fit_predict(df, feats, tr, te, seeds)
        fold_of.iloc[te] = k

    crops = sorted(df["Crop"].unique())
    engine = pd.Series(np.nan, index=df.index)
    work = df[["District", "Season", "Year", "Crop", "relevance"]].copy()
    work["_learned"] = learned
    work["score_rules"] = rules["score_rules"]

    for k, (_tr, te) in enumerate(folds):
        if protocol == "fc":
            # only years already scored by an earlier forward-chaining fold
            earlier = fold_of.between(0, k - 1)
            source = work[earlier]
        else:
            test_districts = set(df.iloc[te]["District"])
            source = work[work["_learned"].notna() & ~work["District"].isin(test_districts)]
        if len(source):
            alphas = blend.alpha_from_skill(blend.per_crop_ranking_skill(source, "_learned"))
        else:
            alphas = {c: 1.0 for c in crops}       # no history yet: the ranker alone
        test = work.iloc[te]
        engine.iloc[te] = blend.served_scores(
            test, "_learned", "score_rules", alphas,
            vetoed=rules["vetoed"].iloc[te].to_numpy() if served else None).to_numpy()

    tested = fold_of >= 0
    out = df.loc[tested, ["District", "Season", "Year", "Crop", "relevance",
                          "area_share"]].copy()
    serv = rules.loc[tested, "servable"] if served else pd.Series(True, index=out.index)
    out["score_engine"] = _sunk(engine[tested], serv)
    out["score_ranker"] = _sunk(learned[tested], serv)

    pop = baselines.popularity_prior_split(df, splitter)
    out["score_popularity"] = _sunk(pop[tested].fillna(0.0), serv)
    out["score_rules"] = _sunk(rules.loc[tested, "score_rules"], serv)
    if protocol == "fc":
        out["score_persistence"] = _sunk(baselines.district_persistence(df)[tested], serv)

    methods = [("Popularity prior (same protocol)", "score_popularity"),
               ("S2 rules only", "score_rules"),
               ("Ranker alone (learned score)", "score_ranker"),
               ("Engine as served (blend + veto)", "score_engine")]
    if protocol == "fc":
        methods.insert(1, ("District persistence (last year's share)", "score_persistence"))
    report = pd.DataFrame([{"Method": name, **ranking_report(out, col)}
                           for name, col in methods])
    paired = district_paired_test(out, "score_popularity", "score_engine")
    return ProtocolResult(protocol, label, out, report, paired)
