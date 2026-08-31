"""The end-to-end recommendation pipeline (plan §3).

Wires the five stages together in the order the plan specifies:

    S0  feature store          -> the taluka's 165 features
    S1  LambdaMART ranker      -> learned crop scores        (learned)
    S2  agronomic gate         -> suitability + hard vetoes  (rules)
    S5  blend + conformal + OOD guard                        (hybrid)
    S3  quantile yield model   -> p10/p50/p90 band           (learned)
    S4  fertiliser engine      -> exact dose plan            (lookup)

**S1 proposes, S2 vetoes.** A ranker trained on 34 districts will occasionally
suggest something agronomically impossible; the rule gate catches that class of
error deterministically, and no amount of model tuning substitutes for it.

**S4 depends on S1's output but not on its confidence.** Once a crop is chosen
— by the model or by the farmer overriding it — the dose is exact. Fertiliser
advice therefore stays correct even when crop advice is uncertain, which is the
failure mode you want.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib

import numpy as np
import pandas as pd

from src import config
from src.data.load import load_shc, taluka_universe
from src.data.training_set import build, district_features, model_features
from src.eval.metrics import per_crop_spearman
from src.features.agronomic_fit import FIT_COLUMNS, compute
from src.features.build_store import build_feature_store
from src.features.panel_history import LAG_COLUMNS
from src.features.build_store import feature_columns as store_feature_columns
from src.models import (blend, conformal, ranker, yield_class, yield_quantile,
                        yield_regimes)
from src.ontology.crop_map import CROP_ONTOLOGY, local_names, to_fertiliser_crop
from src.rules import fertiliser as fz
from src.rules.soil_class import SoilTest, classify, taluka_soil_test
from src.rules.suitability import score as suitability_score
from src.rules.suitability import water_limited

# --------------------------------------------------------------- training ---
@dataclass
class TrainedPipeline:
    ranker_model: object
    quantile_models: dict
    features: list
    alphas: dict
    conformal_q: dict
    guard: conformal.MahalanobisGuard
    guard_features: list
    crop_stats: pd.DataFrame
    leach_quantiles: pd.Series
    rank_skill: pd.DataFrame | None = None
    yield_skill: pd.DataFrame | None = None
    #: per-crop skill under each regime; drives the abstention rule
    crop_skill: pd.DataFrame | None = None
    #: crops the model may emit a yield number for at all
    #: {"cold": frozenset, "warm": frozenset} — servable crops per regime
    yield_servable: dict = field(default_factory=dict)
    #: warm/cold regime models and their feature lists
    regimes: dict | None = None
    #: tercile classifier — the primary S3 output
    class_models: dict | None = None
    #: history, keyed (District, Crop, Season) -> the latest lag feature row
    history: pd.DataFrame | None = None


def train(seeds: int = 3, verbose: bool = True) -> TrainedPipeline:
    """Fit every learned component and the calibration layer."""
    def log(msg):
        if verbose:
            print(msg)

    log("S0  building training frame ...")
    df = attach_fit_features(build())
    # model_features already picks the numeric fit_* columns up; listing them
    # again would hand LightGBM a duplicated feature name
    feats = model_features(df)
    assert len(feats) == len(set(feats))
    assert fit_columns(df), "early-fusion features missing from the training frame"
    y = yield_quantile.training_rows(df)

    log("S1  fitting the ranker ...")
    rank_model = ranker.fit(df, feats)

    log("S3  fitting the two yield regimes and the tercile classifier ...")
    qmodels = yield_quantile.fit_quantiles(y, feats)
    regimes = yield_regimes.fit(df)
    hist = yield_regimes.prepare(df)
    cold_feats, warm_feats = yield_regimes.feature_sets(hist)
    class_models = {
        "cold": yield_class.fit(hist, cold_feats),
        "warm": yield_class.fit(hist[hist["has_history"] == 1], warm_feats),
        "cold_features": cold_feats, "warm_features": warm_feats,
    }

    log("S3  measuring per-crop skill to decide where to abstain ...")
    skill = yield_regimes.crop_skill(df, seeds=seeds)
    # per-regime, because the two regimes answer different queries and a crop
    # can be predictable with history and unpredictable without it
    servable = {
        "cold": frozenset(skill.loc[skill["serve_cold"], "Crop"]),
        "warm": frozenset(skill.loc[skill["serve_warm"], "Crop"]),
    }

    log("S5  measuring per-crop skill for the blend weights ...")
    y = y.copy()
    oof = yield_quantile.cross_val_quantiles(y, feats, seeds=seeds)
    y = y.join(oof)
    # The yield model's per-crop skill governs the *yield band*, and nothing
    # else. Using it to weight the ranker — as the first implementation did —
    # zeroed out cotton, gram, tur and safflower and handed them to the rule
    # scorer, costing 0.098 NDCG@5 (p < 0.0001). The yield model fails on those
    # crops; the ranker does not (cotton rho = 0.68, gram rho = 0.74).
    yield_skill = per_crop_spearman(y, "p50")

    # The alpha weights govern S1, which is a ranker, so they come from
    # ranking skill measured the way the ranker is actually used.
    ranker_oof = df.assign(_learned=ranker.cross_val_scores(df, feats, seeds=seeds))
    rank_skill = blend.per_crop_ranking_skill(ranker_oof, "_learned")
    alphas = blend.alpha_from_skill(rank_skill)

    log("S5  calibrating conformal intervals ...")
    conformal_q = {}
    for alpha in config.CONFORMAL_ALPHAS:
        sc = conformal.SplitConformal(alpha).calibrate(y["yield_z"], y["p10"], y["p90"])
        conformal_q[alpha] = sc.q_

    log("S5  fitting the out-of-distribution guard ...")
    # Fitted on the 351 TALUKAS, not on the 34 district means. The first
    # version was fitted on the same district vectors it then scored, so
    # nothing was ever out-of-distribution and it fired on 0 of 30 queries.
    store = build_feature_store()
    guard_cols = [c for c in store_feature_columns(store)
                  if store[c].notna().all()][:40]
    # quantile=1.0: the cut sits at the largest distance in the served corpus,
    # so a taluka that IS training data never abstains. See the class docstring
    # for the regional-holdout evidence that it still detects real novelty.
    guard = conformal.MahalanobisGuard(quantile=1.0).fit(store[guard_cols])

    crop_stats = (y.groupby("Crop")[["crop_yield_mean", "crop_yield_std"]]
                    .first().reset_index())
    store = build_feature_store()
    leach_q = store["leach_risk"].rank(pct=True)
    leach_q.index = pd.MultiIndex.from_frame(store[config.KEY])

    # the most recent history row per (district, crop, season), for warm serving
    latest = (hist.sort_values("Year")
                  .groupby(["District", "Crop", "Season"], as_index=False).last())

    return TrainedPipeline(rank_model, qmodels, feats, alphas, conformal_q,
                           guard, guard_cols, crop_stats, leach_q,
                           rank_skill=rank_skill, yield_skill=yield_skill,
                           crop_skill=skill, yield_servable=servable,
                           regimes=regimes, class_models=class_models,
                           history=latest)


def _cache_key() -> str:
    """Fingerprint of everything the fitted pipeline depends on.

    The feature store plus the source of every module that shapes the models.
    Any edit to those invalidates the cache, so a stale pipeline can never be
    served after a code or data change.
    """
    h = hashlib.sha256()
    store = config.FEATURES / "taluka_features.parquet"
    if store.exists():
        h.update(store.read_bytes())
    for mod in ("config.py", "data/training_set.py", "models/ranker.py",
                "models/yield_quantile.py", "models/blend.py",
                "models/conformal.py", "features/build_store.py",
                "rules/suitability.py", "rules/crop_requirements.py"):
        f = Path(__file__).parent / mod
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def load_pipeline(use_cache: bool = True) -> TrainedPipeline:
    """Fit once, then reuse — in-process and across processes.

    Fitting takes ~60 s, which the CLI would otherwise pay on every single
    invocation. The cache is keyed on a hash of the feature store and of every
    module that shapes the models, so it cannot serve a stale pipeline.
    """
    path = config.ARTIFACTS / f"pipeline_{_cache_key()}.joblib"
    if use_cache and path.exists():
        try:
            return joblib.load(path)
        except Exception:
            path.unlink(missing_ok=True)      # corrupt or version-skewed

    pipe = train(verbose=False)
    if use_cache:
        for stale in config.ARTIFACTS.glob("pipeline_*.joblib"):
            stale.unlink(missing_ok=True)
        joblib.dump(pipe, path, compress=3)
    return pipe


# ---------------------------------------------------------------- serving ---
@dataclass
class CropAdvice:
    crop: str
    crop_marathi: str | None
    rank: int
    final_score: float
    learned_score: float | None
    rule_score: float | None
    suitability_class: str
    limiting_factor: str
    decided_by: str
    reason: str
    yield_p10_t_ha: float | None = None
    yield_p50_t_ha: float | None = None
    yield_p90_t_ha: float | None = None
    yield_interval_note: str = ""
    #: "below" / "typical" / "above" the crop's norm — the primary S3 output
    yield_class: str | None = None
    yield_class_confidence: float | None = None
    #: which regime answered, and why
    yield_regime: str | None = None
    yield_abstained: bool = False
    #: the eight Liebig factor scores, 0-1. The gate's score is their minimum,
    #: so exposing them lets a client show *why* — which stave is short.
    factors: dict = field(default_factory=dict)
    requires_irrigation: bool = False
    evidence: dict = field(default_factory=dict)
    fertiliser: dict | None = None


@dataclass
class Recommendation:
    district: str
    taluka: str
    season: str
    irrigated: bool
    soil_class: str
    soil_test_source: str
    confident: bool
    novelty: float | None
    abstention_reason: str | None
    water_limited: bool
    crops: list = field(default_factory=list)
    vetoed: list = field(default_factory=list)
    not_assessable: list = field(default_factory=list)
    micronutrients: list = field(default_factory=list)
    context: dict = field(default_factory=dict)

    def to_json(self, **kw) -> str:
        return json.dumps(asdict(self), default=float, ensure_ascii=False, **kw)


def fit_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in FIT_COLUMNS if c in df.columns]


def attach_fit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Early fusion: S2's per-factor scores become S1 features.

    S2 reads only soil and climate, never a label, so this is leakage-free by
    construction. Measured at +0.035 NDCG@5 (p = 0.0002) inside the full
    engine: the fit features raise the ranker's per-crop skill, which raises
    the alpha weights, which in turn reduces how much the weak rule scorer is
    leaned on. The effect compounds through the blend, which is why it is
    larger here than on the bare ranker (+0.005, not significant on its own).
    """
    return pd.concat([df, compute(df, district_features())], axis=1)


@lru_cache(maxsize=1)
def _store() -> pd.DataFrame:
    return build_feature_store()


@lru_cache(maxsize=1)
def _district_feats() -> pd.DataFrame:
    return district_features().set_index("District")


def taluka_row(district: str, taluka: str) -> dict:
    s = _store()
    hit = s[(s["District"] == district.upper()) & (s["Taluka"] == taluka.upper())]
    if hit.empty:
        raise KeyError(f"{district}/{taluka} is not one of the 351 Soil Health Card talukas")
    return hit.iloc[0].to_dict()


def recommend(
    district: str,
    taluka: str,
    season: str,
    *,
    soil_test: SoilTest | None = None,
    irrigated: bool = False,
    top_k: int = 5,
    pipe: TrainedPipeline | None = None,
) -> Recommendation:
    """Full five-stage recommendation for one taluka and season."""
    pipe = pipe or load_pipeline()
    district, taluka = district.upper(), taluka.upper()
    feats = taluka_row(district, taluka)

    # ---- S4 layer 1: which fertility class is this land in? ----------------
    test = soil_test or taluka_soil_test(feats)
    soil_class = classify(test)
    source = "farmer soil health card" if soil_test else "taluka SHC distribution"

    # ---- S5 out-of-distribution guard --------------------------------------
    # Scored on THIS taluka against the 351-taluka distribution, so an unusual
    # place genuinely trips it.
    missing = [c for c in pipe.guard_features if c not in feats]
    if missing:
        ood, novelty = True, float("nan")
    else:
        x = pd.DataFrame([{c: feats[c] for c in pipe.guard_features}])
        ood = bool(pipe.guard.is_outlier(x)[0])
        novelty = float(pipe.guard.novelty(x)[0])

    # ---- S1 learned scores over the full candidate set ---------------------
    candidates = list(CROP_ONTOLOGY) + sorted(
        c for c in _apy_crops() if c not in CROP_ONTOLOGY
    )
    query = _query_frame(district, season, candidates, pipe)
    learned = pd.Series(pipe.ranker_model.predict(query[pipe.features]), index=candidates)

    # ---- S2 rule scores and hard vetoes ------------------------------------
    rule_scores, suit = {}, {}
    for apy_crop in candidates:
        fert_crop = to_fertiliser_crop(apy_crop)
        if fert_crop is None:
            rule_scores[apy_crop], suit[apy_crop] = np.nan, None
            continue
        s = suitability_score(fert_crop, season, feats, irrigated=irrigated)
        suit[apy_crop] = s
        rule_scores[apy_crop] = np.nan if not s.scored else s.score

    frame = pd.DataFrame({
        "Crop": candidates,
        "learned": learned.to_numpy(),
        "rules": pd.Series(rule_scores).reindex(candidates).to_numpy(),
        "District": district,
        "Season": season,
    })
    # crops the gate cannot assess keep the median rule score rather than being
    # penalised for the gate's own ignorance
    frame["rules"] = frame["rules"].fillna(frame["rules"].median())

    # ---- S5 blend, or fall back to rules alone when out of distribution ----
    frame["final"] = blend.blend(frame, "learned", "rules", pipe.alphas,
                                 ood_mask=np.full(len(frame), ood))
    frame = frame.sort_values("final", ascending=False)

    # ---- apply the hard veto AFTER scoring, and log every one --------------
    vetoed = []
    keep = []
    for row in frame.itertuples(index=False):
        s = suit[row.Crop]
        if s is not None and s.scored and s.vetoed:
            vetoed.append({"crop": row.Crop, "limiting_factor": s.limiting_factor,
                           "reason": s.reason, "score": s.score})
        else:
            keep.append(row)

    # A crop with neither an agronomic envelope nor a fertiliser recipe cannot
    # be assessed or acted on — the seven APY aggregates ("Other Kharif
    # pulses") and tobacco. Listing them as recommendations would be padding
    # the answer with rows the system has nothing to say about, so they are
    # reported separately instead of competing for a top-k slot.
    not_assessable, recommendable = [], []
    for row in keep:
        s_row = suit[row.Crop]
        assessable = s_row is not None and s_row.scored
        has_recipe = to_fertiliser_crop(row.Crop) is not None
        if assessable or has_recipe:
            recommendable.append(row)
        else:
            not_assessable.append({
                "crop": row.Crop,
                "reason": "No agronomic envelope and no fertiliser recipe — "
                          "an APY aggregate or an unmodelled crop.",
            })
    keep = recommendable

    advice = []
    for rank, row in enumerate(keep[:top_k], start=1):
        advice.append(_build_advice(rank, row, suit, feats, district, taluka,
                                    test, soil_class, season, pipe, ood))

    return Recommendation(
        district=district, taluka=taluka, season=season, irrigated=irrigated,
        soil_class=soil_class, soil_test_source=source,
        confident=not ood,
        novelty=round(novelty, 2) if novelty == novelty else None,
        abstention_reason=(
            f"This taluka sits outside the distribution of the 351 talukas the "
            f"model was fitted on (Mahalanobis novelty {novelty:.2f}x the "
            f"threshold). The learned ranker is suppressed and these "
            f"recommendations come from the agronomic rule scorer alone."
            if ood else None),
        water_limited=water_limited(feats, season),
        crops=advice,
        vetoed=vetoed,
        not_assessable=not_assessable,
        micronutrients=fz.micronutrient_plan(feats),
        context=_context(feats, season, test),
    )


def _build_advice(rank, row, suit, feats, district, taluka, test, soil_class,
                  season, pipe, ood) -> CropAdvice:
    s = suit[row.Crop]
    alpha = pipe.alphas.get(row.Crop, 0.0)
    decided = "rules (out of distribution)" if ood else (
        "rules" if alpha == 0 else "model" if alpha >= 0.8 else "blend")

    fert_crop = to_fertiliser_crop(row.Crop)
    adv = CropAdvice(
        crop=row.Crop,
        crop_marathi=local_names().get(fert_crop) if fert_crop else None,
        rank=rank,
        final_score=round(float(row.final), 3),
        learned_score=None if ood else round(float(row.learned), 3),
        rule_score=round(float(row.rules), 3),
        suitability_class=s.suitability_class if s else "?",
        limiting_factor=s.limiting_factor if s else "unknown",
        decided_by=decided,
        reason=(s.reason if s else "No agronomic envelope encoded; "
                                   "ranked on learned preference alone."),
        factors=dict(s.factors) if s and s.factors else {},
        requires_irrigation=bool(s.requires_irrigation) if s else False,
        evidence=dict(s.evidence) if s and s.evidence else {},
    )

    # ---- S3 yield: class first, band only where the model has skill --------
    _attach_yield(adv, row.Crop, district, season, pipe)

    # ---- S4 fertiliser ------------------------------------------------------
    if fert_crop:
        adv.fertiliser = _fertiliser_plan(district, taluka, fert_crop, feats, test, pipe)
    return adv


def _attach_yield(adv, apy_crop: str, district: str, season: str, pipe) -> None:
    """S3 output: tercile class as primary, band only where skill justifies it.

    Three decisions, in order:

    1. **Which regime?** A district-crop-season with history gets the warm
       model and its lag features; one without gets the cold model. These are
       different products and the answer says which one replied.
    2. **Class before number.** "Below / typical / above this crop's norm" is a
       question the data answers at 52% against a 34% baseline. A point yield
       is not, so the class leads.
    3. **Abstain where there is no skill.** If cross-validated per-crop rho is
       below the floor, no yield number is emitted at all — the same
       philosophy S5 already applies to out-of-distribution talukas.
    """
    if pipe.regimes is None or pipe.class_models is None:
        return

    hist_row = _history_row(apy_crop, district, season, pipe)
    warm = hist_row is not None
    adv.yield_regime = "warm (district has history)" if warm else "cold (no history)"

    query = _yield_query(apy_crop, district, season, pipe, hist_row)
    if query is None:
        return

    key = "warm" if warm else "cold"
    feats = pipe.class_models[f"{key}_features"]
    clf = pipe.class_models[key]
    proba = clf.predict_proba(query[feats])[0]
    idx = int(np.argmax(proba))
    adv.yield_class = yield_class.CLASSES[idx]
    adv.yield_class_confidence = round(float(proba[idx]), 3)

    servable = pipe.yield_servable.get(key, frozenset()) \
        if isinstance(pipe.yield_servable, dict) else pipe.yield_servable
    if apy_crop not in servable:
        adv.yield_abstained = True
        adv.yield_interval_note = (
            f"No yield range offered: cross-validated skill for {apy_crop} in "
            f"the {key}-start regime is below the threshold, so any number "
            f"would be noise. The class above is the most this data supports "
            f"for this crop here."
        )
        return

    band = _yield_band(apy_crop, district, season, pipe, query, key)
    if band:
        adv.yield_p10_t_ha, adv.yield_p50_t_ha, adv.yield_p90_t_ha = band
        adv.yield_interval_note = (
            "Conformally calibrated 90% band; empirical coverage 0.900 on "
            "held-out districts over eight years. Reported as a range because "
            "the honest within-crop signal is rho = 0.50 even at best — a "
            "point estimate would mislead."
        )


def _history_row(apy_crop, district, season, pipe):
    h = pipe.history
    if h is None:
        return None
    hit = h[(h["District"] == district) & (h["Crop"] == apy_crop)
            & (h["Season"] == season)]
    return hit.iloc[0] if not hit.empty else None


def _yield_query(apy_crop, district, season, pipe, hist_row):
    """One-row frame carrying the features the chosen regime expects."""
    q = _query_frame(district, season, [apy_crop], pipe).copy()
    for col in LAG_COLUMNS:
        q[col] = float(hist_row[col]) if (hist_row is not None
                                          and col in hist_row
                                          and pd.notna(hist_row[col])) else np.nan
    if hist_row is None:
        q["has_history"] = 0
        q["n_prior_years"] = 0
    return q


def _yield_band(apy_crop, district, season, pipe, query=None, regime="cold") -> tuple | None:
    stats = pipe.crop_stats
    hit = stats[stats["Crop"] == apy_crop]
    if hit.empty:
        return None
    mean, std = float(hit["crop_yield_mean"].iat[0]), float(hit["crop_yield_std"].iat[0])
    # A crop observed in too few districts has no estimable spread (tobacco is
    # recorded once in all of Maharashtra). Reporting a band there would be
    # inventing a number, so no band is offered at all.
    if not np.isfinite(std) or std <= 0:
        return None
    q = _query_frame(district, season, [apy_crop], pipe) if query is None else query
    models = pipe.regimes[f"{regime}_quantiles"] if pipe.regimes else pipe.quantile_models
    feats = pipe.regimes[f"{regime}_features"] if pipe.regimes else pipe.features
    zs = {a: float(m.predict(q[feats])[0]) for a, m in models.items()}
    widen = pipe.conformal_q.get(0.1, 0.0)
    lo = min(zs.values()) - widen
    hi = max(zs.values()) + widen
    mid = zs[0.5]
    band = [yield_quantile.to_tonnes_per_ha(v, mean, std) for v in (lo, mid, hi)]
    return tuple(round(max(0.0, float(b)), 2) for b in band)


def _fertiliser_plan(district, taluka, fert_crop, feats, test, pipe) -> dict | None:
    from src.rules.cost_optimiser import compare_with_table, optimise

    pct = float(pipe.leach_quantiles.get((district, taluka), 0.5))
    rec = fz.recommend(district, fert_crop, feats, soil_test=test, leach_percentile=pct)
    if rec is None:
        return _state_median_plan(fert_crop, feats, test, pct)

    opts = {}
    for opt in (1, 2):
        hit = fz.lookup(district, fert_crop, rec.soil_class, option=opt)
        if hit:
            opts[opt] = hit["products"]

    # Two distinct LP questions, and conflating them produces a nonsense
    # "saving": the published options deliver the *table's* target, so the
    # saving must be measured against that same target. The plan the farmer
    # actually gets is the LP on the *interpolated* target, which is a
    # different (usually larger) dose and therefore not comparable on price.
    saving = compare_with_table(opts, rec.exact_table["target"]) if opts else None
    needs_sulphur = any(m["component"] == "S" for m in rec.micronutrients)
    plan = optimise(
        rec.interpolated_target,
        require_sulphur_kg=20.0 if needs_sulphur else 0.0,
    )

    return {
        "available": True,
        "soil_class": rec.soil_class,
        "table_products_kg_ha": rec.exact_table["products"],
        "table_target_kg_ha": rec.exact_table["target"],
        "interpolated_target_kg_ha": rec.interpolated_target,
        "nitrogen_schedule": rec.schedule,
        "sulphur_swap": rec.sulphur_swap,
        "recommended_mix": plan,
        "least_cost_vs_table": saving,
        "notes": rec.notes,
    }


def _state_median_plan(fert_crop, feats, test, pct) -> dict:
    """Fallback for the 36% of (crop, district) cells the table omits."""
    from src.rules.cost_optimiser import optimise

    soil_class = classify(test)
    hit = fz.state_median_lookup(fert_crop, soil_class)
    if hit is None:
        return {"available": False, "estimated": False,
                "note": f"No fertiliser recommendation exists for {fert_crop} "
                        f"anywhere in the table."}

    targets = fz.crop_targets_state(fert_crop)
    target = (fz.interpolate_target(targets, test) if len(targets) == 3
              else dict(hit["target"]))
    target = {k: round(v, 1) for k, v in target.items() if v is not None}

    micro = fz.micronutrient_plan(feats)
    return {
        "available": True,
        "estimated": True,
        "soil_class": soil_class,
        "table_products_kg_ha": {k: round(v, 2) for k, v in hit["products"].items()},
        "product_ranges_across_districts": hit["product_ranges"],
        "table_target_kg_ha": hit["target"],
        "interpolated_target_kg_ha": target,
        "nitrogen_schedule": fz.split_schedule(target.get("N", 0.0),
                                               feats.get("leach_risk", 0.0), pct),
        "sulphur_swap": fz.sulphur_substitution(hit["products"], micro),
        "recommended_mix": optimise(
            target, require_sulphur_kg=20.0 if any(m["component"] == "S" for m in micro) else 0.0),
        "least_cost_vs_table": None,
        "notes": [hit["note"]],
    }


@lru_cache(maxsize=1)
def _apy_crops() -> tuple:
    from src.data.load import load_apy
    return tuple(sorted(load_apy()["Crop"].unique()))


@lru_cache(maxsize=128)
def _query_frame_cached(district: str, season: str, crops: tuple,
                        feature_key: str) -> pd.DataFrame:
    df = build()
    template = df[(df["District"] == district) & (df["Season"] == season)]
    if template.empty:
        template = df[df["Season"] == season]
    rows = []
    for crop in crops:
        hit = template[template["Crop"] == crop]
        rows.append((hit.iloc[0] if not hit.empty else template.iloc[0]).copy())
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["Crop"] = list(crops)
    out["District"] = district
    out["Season"] = season
    # the serving frame must carry exactly the features the model was fitted on
    return attach_fit_features(out)


def _query_frame(district: str, season: str, crops: list, pipe) -> pd.DataFrame:
    # Keyed on the pipeline's feature signature, not on id(pipe): CPython
    # reuses object ids after garbage collection, so a retrained pipeline could
    # silently inherit a previous one's cached query frames.
    key = hashlib.sha256("|".join(pipe.features).encode()).hexdigest()[:16]
    return _query_frame_cached(district, season, tuple(crops), key)


def _context(feats: dict, season: str, test: SoilTest) -> dict:
    return {
        "annual_rainfall_mm": round(float(feats["rain_annual"]), 0),
        "season_rainfall_mm": round(float(feats.get(f"rain_{season}", np.nan)), 0)
        if season in config.SEASONS else None,
        "rootzone_awc_mm": round(float(feats["rootzone_awc"]), 0),
        "soil_depth_mm": float(feats["depth_mm"]),
        "drainage_ord": float(feats["drainage_ord"]),
        "aridity_index": round(float(feats["aridity_annual"]), 2),
        "lgp_days": float(feats["lgp_annual"]),
        "leach_risk": round(float(feats["leach_risk"]), 1),
        "soil_test": {k: (round(v, 1) if v is not None else None)
                      for k, v in test.as_dict().items()},
        "shc_samples": int(feats["n_samples"]),
    }
