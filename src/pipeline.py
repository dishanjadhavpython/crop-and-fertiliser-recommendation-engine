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
from dataclasses import asdict, dataclass, field, replace
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
from src.rules import soil_fusion
from src.rules.crop_requirements import get as crop_requirement
from src.rules.soil_class import SoilTest, classify, taluka_soil_test
from src.rules.suitability import score as suitability_score
from src.rules.suitability import water_limited

#: Feature families the out-of-distribution guard watches. Climate first,
#: because that is what separates the Konkan from Marathwada; then soil
#: physics; then the nutrient distribution.
GUARD_PREFIXES = ("rain_", "aridity_", "lgp_", "cwb_", "gdd_", "tmax_mean_",
                  "tmin_mean_", "dry_spell_", "rootzone_awc", "awc_mm_m",
                  "depth_mm", "drainage_ord", "texture_ord", "ph_class_value",
                  "ec_saline", "NI_", "def_", "leach_risk", "drought_vuln",
                  "water_supply")


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
    rank_model = ranker.fit(df, feats, seeds=config.RANKER_SERVED_SEEDS, monotone=True)

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

    log("S5  calibrating conformal intervals, per regime ...")
    from src.eval.splits import forward_chaining, group_kfold
    conformal_q = {}
    for regime, frame, rfeats, splitter in (
            ("cold", hist, cold_feats, group_kfold),
            ("warm", hist[hist["has_history"] == 1].reset_index(drop=True),
             warm_feats, forward_chaining)):
        q = yield_regimes.oof_quantiles(frame, rfeats, splitter)
        m = frame.join(q).dropna(subset=["p10", "p90"])
        for alpha in config.CONFORMAL_ALPHAS:
            sc = conformal.SplitConformal(alpha).calibrate(m["yield_z"], m["p10"], m["p90"])
            conformal_q[(regime, alpha)] = sc.q_
    # the scalar keys stay, so anything still reading conformal_q[0.1] gets the
    # cold-start widening rather than a KeyError
    for alpha in config.CONFORMAL_ALPHAS:
        conformal_q[alpha] = conformal_q[("cold", alpha)]

    log("S5  fitting the out-of-distribution guard ...")
    # Fitted on the 351 TALUKAS, not on the 34 district means. The first
    # version was fitted on the same district vectors it then scored, so
    # nothing was ever out-of-distribution and it fired on 0 of 30 queries.
    store = build_feature_store()
    # Explicit, and spanning what the query actually varies over: climate, soil
    # physics and soil chemistry. The first version took "the first 40 complete
    # columns", which happened to be Soil Health Card nutrient distributions
    # only — so the guard could not see a climatically unusual taluka at all,
    # while a farmer typing EC = high wrote 100 into one of the columns it did
    # watch and tripped it. Farmer readings are range-checked at the API
    # instead; the guard is scored on the taluka as surveyed.
    guard_cols = [c for c in store_feature_columns(store)
                  if store[c].notna().all()
                  and c.startswith(GUARD_PREFIXES)][:60]
    # quantile=1.0: the cut sits at the largest distance in the served corpus,
    # so a taluka that IS training data never abstains. See the class docstring
    # for the regional-holdout evidence that it still detects real novelty.
    guard = conformal.MahalanobisGuard(quantile=1.0).fit(store[guard_cols])

    # The z-score is taken within (Crop, Year), so converting a prediction back
    # to t/ha needs a year's statistics. ``.first()`` took 2015-16 — the oldest
    # year in the panel — so every yield a farmer read was expressed on a scale
    # eight years out of date. The most recent three years are averaged instead,
    # falling back to the whole panel for a crop absent from them.
    stats_cols = ["crop_yield_mean", "crop_yield_std"]
    overall = y.groupby("Crop")[stats_cols].mean()
    recent_years = sorted(y["Year"].unique())[-3:]
    recent = y[y["Year"].isin(recent_years)].groupby("Crop")[stats_cols].mean()
    crop_stats = recent.reindex(overall.index).fillna(overall).reset_index()
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


def cache_inputs() -> list[Path]:
    """Every file the fitted pipeline depends on.

    The first version hashed ``taluka_features.parquet`` — which serving never
    reads, because ``build_feature_store()`` rebuilds from the raw CSVs — plus
    nine hand-listed modules. Everything else could change without changing the
    key: the raw data itself, ``pipeline.py``, and the feature blocks. A stale
    pipeline could therefore be served after a real change, which is the one
    thing a cache key exists to prevent. ``scorecard.py`` gates on this list.
    """
    src = Path(__file__).resolve().parent
    raw = [p for p in config.RAW.iterdir() if p.suffix in (".csv", ".json")]
    extra = [p for p in (config.ARTIFACTS / "model_selection.json",) if p.exists()]
    return sorted(set(raw) | set(src.rglob("*.py")) | set(extra))


def _cache_key() -> str:
    """Fingerprint of everything in ``cache_inputs()``, plus the library versions.

    Source is hashed by content. Raw data is hashed by name, size and
    modification time instead: the weather panel alone is ~230 MB and serving
    would otherwise re-read all of it on every start. The failure that would
    need — editing a CSV in place without changing its size or its mtime — is
    not one that happens by accident, and re-copying a file invalidates the key
    in the safe direction.
    """
    import lightgbm
    import sklearn

    h = hashlib.sha256()
    for f in cache_inputs():
        h.update(str(f.name).encode())
        if f.suffix == ".py":
            h.update(f.read_bytes())
        else:
            stat = f.stat()
            h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    h.update(f"lightgbm{lightgbm.__version__}sklearn{sklearn.__version__}".encode())
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
    soil_photo: dict[str, float] | None = None,
    photo_in_distribution: bool = True,
    top_k: int = 5,
    pipe: TrainedPipeline | None = None,
) -> Recommendation:
    """Full five-stage recommendation for one taluka and season."""
    pipe = pipe or load_pipeline()
    district, taluka = district.upper(), taluka.upper()
    feats = taluka_row(district, taluka)
    # The taluka as surveyed, kept before any farmer reading is substituted in.
    # The out-of-distribution guard is scored on this: whether this *place* is
    # unlike the places the model learned from is not a question a farmer's own
    # pH or EC reading can answer.
    surveyed = dict(feats)

    # ---- the farmer's photograph, weighed against the survey ---------------
    # `soil_photo` is the classifier's calibrated probabilities over its own
    # classes. The survey is the prior and the photograph the evidence; the
    # fusion may move texture and available water, and may never touch depth,
    # drainage or salinity, so a photograph can add a constraint but never lift
    # a veto. It is computed against `surveyed` above, not the fused features,
    # for the same reason the OOD guard is.
    fusion = soil_fusion.fuse(
        _surveyed_soil(district, taluka), soil_photo, config.F_SOIL_MODEL_META,
        in_distribution=photo_in_distribution)
    feats = soil_fusion.apply_to_features(feats, fusion)

    # ---- S4 layer 1: which fertility class is this land in? ----------------
    # Field by field, because a card is often read in part. A missing nutrient
    # used to fall through to the Medium archetype inside the interpolation —
    # a different field's dose — rather than to this taluka's own value, and
    # the answer then claimed to come from the farmer's card regardless.
    taluka_test = taluka_soil_test(feats)
    sources: dict[str, str] = {}
    if soil_test is None:
        test = taluka_test
        for key in ("N", "P", "K", "OC"):
            sources[key] = "taluka SHC distribution"
    else:
        filled = {}
        for attr, key in (("n_kg_ha", "N"), ("p_kg_ha", "P"),
                          ("k_kg_ha", "K"), ("oc_pct", "OC")):
            given = getattr(soil_test, attr)
            filled[attr] = given if given is not None else getattr(taluka_test, attr)
            sources[key] = ("farmer soil health card" if given is not None
                            else "taluka SHC distribution")
        test = replace(soil_test, **filled)

    given_ph = soil_test is not None and soil_test.ph is not None
    given_ec = soil_test is not None and soil_test.ec_status is not None
    farmer_micro = (soil_test.micronutrients or {}) if soil_test is not None else {}
    sources["pH"] = ("farmer soil health card" if given_ph
                     else "taluka soil survey pH class")
    sources["EC"] = ("farmer soil health card" if given_ec
                     else "taluka SHC distribution")
    for comp in config.MICRONUTRIENTS:
        sources[comp] = ("farmer soil health card" if farmer_micro.get(comp)
                         else "taluka SHC distribution")

    soil_class = classify(test)
    from_card = sum(1 for v in sources.values() if v == "farmer soil health card")
    source = ("taluka SHC distribution" if from_card == 0
              else "farmer soil health card" if from_card == len(sources)
              else f"farmer soil health card ({from_card} of {len(sources)} readings; "
                   f"the rest from the taluka)")

    # The farmer's own pH/EC reading, when given, replaces the taluka average
    # for every rule downstream (S2's Liebig gate, S5's OOD guard) — the same
    # "a real reading from this field outranks a taluka-wide average"
    # principle as the N/P/K/OC override above, applied to the two SHC
    # components that feed rules rather than the fertility-class table.
    if test.ph is not None:
        feats["ph_class_value"] = test.ph
    # EC is asymmetric, because the card's verdict and the survey's statistic
    # are not the same quantity. `ec_saline` is the share of the taluka's
    # samples in the survey's *saline class* — a rare class: the statewide
    # median share is 0.1% and the maximum 39%. The card's "high" only says the
    # reading is above the range printed on that card (0.2-0.9 dS/m on the
    # fixture card, which reads 1.06). That is far short of the saline class,
    # so it cannot be substituted as "100% of samples saline": doing so zeroed
    # the salinity factor for every crop in the table, sorghum and safflower
    # included — the two the table itself calls salt-tolerant — and a farmer
    # whose EC was a whisker over range was told nothing at all could be grown.
    #
    # "normal" or "low" does settle it: a reading inside the card's range is
    # certainly below the saline class, so this field is not saline. "high"
    # does not, so the survey's share stays as the best estimate and the
    # reading travels to the farmer as a caution (see `_salinity_caution`).
    if test.ec_status is not None and test.ec_status != "high":
        feats["ec_saline"] = 0.0

    # ---- S5 out-of-distribution guard --------------------------------------
    # Scored on THIS taluka as surveyed, against the 351-taluka distribution.
    # The snapshot is taken before the farmer's readings are substituted in:
    # the question the guard answers is "is this place unlike the places the
    # model learned from", and a farmer's own pH is not evidence about that.
    guard_feats = {c: surveyed[c] for c in pipe.guard_features if c in surveyed}
    missing = [c for c in pipe.guard_features if c not in surveyed]
    if missing:
        ood, novelty = True, float("nan")
    else:
        x = pd.DataFrame([guard_feats])
        ood = bool(pipe.guard.is_outlier(x)[0])
        novelty = float(pipe.guard.novelty(x)[0])

    # ---- S1 learned scores over the full candidate set ---------------------
    candidates = list(CROP_ONTOLOGY) + sorted(
        c for c in _apy_crops() if c not in CROP_ONTOLOGY
    )
    query = _taluka_fit(_query_frame(district, season, candidates, pipe),
                        district, feats, irrigated)
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
                                    test, soil_class, season, pipe, ood, irrigated))

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
        water_limited=water_limited(feats, season, irrigated=irrigated),
        crops=advice,
        vetoed=vetoed,
        not_assessable=not_assessable,
        micronutrients=fz.micronutrient_plan(feats, farmer_micro=test.micronutrients),
        context={**_context(feats, season, test, sources, fusion),
                 **_salinity_caution(test, advice)},
    )


#: The crop table's own salt-tolerant group starts here: its notes call sorghum
#: (25) salt-tolerant, safflower (30) very salt-tolerant, and mustard (20) more
#: tolerant than most oilseeds. Below it are the crops a saline reading should
#: make a farmer think twice about.
SALT_TOLERANT_PCT = 20.0


def _salinity_caution(test: SoilTest, advice: list) -> dict:
    """What a card reading of EC "high" means for this farmer's own list.

    Not a veto — the reading cannot place the field in the survey's saline class
    (see the EC note in `recommend`). It is still this field's own measurement,
    so it is reported, together with which of the ranked crops the requirement
    table rates least able to take salt.
    """
    if test.ec_status != "high":
        return {"ec_card_high": False, "ec_least_tolerant": []}
    least = []
    for item in advice:
        # The table is keyed by agronomic name (Chickpea), the list by the
        # market name (Gram); the same bridge the gate itself crosses.
        req = crop_requirement(to_fertiliser_crop(item.crop) or item.crop)
        if req is not None and req.max_saline_pct < SALT_TOLERANT_PCT:
            least.append(item.crop)
    return {"ec_card_high": True, "ec_least_tolerant": least}


def _build_advice(rank, row, suit, feats, district, taluka, test, soil_class,
                  season, pipe, ood, irrigated=False) -> CropAdvice:
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
        adv.fertiliser = _fertiliser_plan(district, taluka, fert_crop, feats, test,
                                          pipe, season, irrigated)
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

    band = _yield_band(apy_crop, district, season, pipe, query, key)  # widened per regime
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
    # the widening measured on THIS regime's own out-of-fold predictions
    widen = pipe.conformal_q.get((regime, 0.1), pipe.conformal_q.get(0.1, 0.0))
    lo = min(zs.values()) - widen
    hi = max(zs.values()) + widen
    mid = zs[0.5]
    band = [yield_quantile.to_tonnes_per_ha(v, mean, std) for v in (lo, mid, hi)]
    return tuple(round(max(0.0, float(b)), 2) for b in band)


def _fertiliser_plan(district, taluka, fert_crop, feats, test, pipe,
                     season=None, irrigated=False) -> dict | None:
    """The dose for THIS farmer's season and water regime.

    The table publishes several recipes for one crop in one district and they
    differ materially; serving used to ask for none of them, so an irrigated
    Rabi query could be answered with the rainfed Kharif dose.
    """
    from src.rules.cost_optimiser import compare_with_table, optimise

    pct = float(pipe.leach_quantiles.get((district, taluka), 0.5))
    rec = fz.recommend(district, fert_crop, feats, soil_test=test, leach_percentile=pct,
                       want_season=season, want_irrigated=irrigated)
    if rec is None:
        return _state_median_plan(fert_crop, feats, test, pct, season, irrigated)

    opts = {}
    for opt in (1, 2):
        hit = fz.lookup(district, fert_crop, rec.soil_class, option=opt,
                        want_season=season, want_irrigated=irrigated)
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
        # which published recipe this is, so the answer can be checked against
        # the table rather than taken on trust
        "context": rec.context,
        "requested": {"season": season, "irrigated": bool(irrigated)},
        "table_products_kg_ha": rec.exact_table["products"],
        "table_target_kg_ha": rec.exact_table["target"],
        "interpolated_target_kg_ha": rec.interpolated_target,
        "nitrogen_schedule": rec.schedule,
        "sulphur_swap": rec.sulphur_swap,
        "recommended_mix": plan,
        "least_cost_vs_table": saving,
        "notes": rec.notes,
    }


def _state_median_plan(fert_crop, feats, test, pct, season=None, irrigated=False) -> dict:
    """Fallback for the 36% of (crop, district) cells the table omits."""
    from src.rules.cost_optimiser import optimise

    soil_class = classify(test)
    hit = fz.state_median_lookup(fert_crop, soil_class, want_season=season,
                                 want_irrigated=irrigated)
    if hit is None:
        return {"available": False, "estimated": False,
                "note": f"No fertiliser recommendation exists for {fert_crop} "
                        f"anywhere in the table."}

    targets = fz.crop_targets_state(fert_crop, want_season=season,
                                    want_irrigated=irrigated)
    target = (fz.interpolate_target(targets, test) if len(targets) == 3
              else dict(hit["target"]))
    target = {k: round(v, 1) for k, v in target.items() if v is not None}

    micro = fz.micronutrient_plan(feats, farmer_micro=test.micronutrients)
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
    # The latest year, not the first. Rows arrive oldest-first, so ``iloc[0]``
    # served every farmer the 2015-16 row — its year_index, its lagged history
    # and its crop statistics — for a question about the coming season.
    newest = template["Year"].max()
    latest = template[template["Year"] == newest]
    rows = []
    for crop in crops:
        hit = latest[latest["Crop"] == crop]
        if hit.empty:
            hit = template[template["Crop"] == crop].sort_values("Year").tail(1)
        rows.append((hit.iloc[0] if not hit.empty else latest.iloc[0]).copy())
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["Crop"] = list(crops)
    out["District"] = district
    out["Season"] = season
    # the serving frame must carry exactly the features the model was fitted on
    return attach_fit_features(out)


def _taluka_fit(query: pd.DataFrame, district: str, feats: dict,
                irrigated: bool) -> pd.DataFrame:
    """Recompute the agronomic-fit block from THIS taluka and THIS farmer.

    The ranker's other features are district properties, and have to be: the
    labels are district-level, so nothing finer can be learned from them. The
    fit block is different in kind — it is *computed*, not learned, S2 scoring a
    crop against land — so at serving there is no reason to evaluate it on a
    district average when the farmer's own taluka is known, with their own pH
    and EC substituted and their own water regime assumed. The same quantity,
    at a finer resolution.

    This is what lets a card change the ranking at all. Before it, the twelve
    readings reached the gate and the dose but never the learned score, so two
    farmers in one district with different soil got the same ranked list.
    """
    source = pd.DataFrame([{**feats, "District": district}]).set_index("District")
    fit = compute(query, source, key="District", irrigated=irrigated)
    out = query.copy()          # _query_frame is lru_cached: never mutate it
    for col in fit.columns:
        out[col] = fit[col].to_numpy()
    return out


def _query_frame(district: str, season: str, crops: list, pipe) -> pd.DataFrame:
    # Keyed on the pipeline's feature signature, not on id(pipe): CPython
    # reuses object ids after garbage collection, so a retrained pipeline could
    # silently inherit a previous one's cached query frames.
    key = hashlib.sha256("|".join(pipe.features).encode()).hexdigest()[:16]
    return _query_frame_cached(district, season, tuple(crops), key)


def _surveyed_soil(district: str, taluka: str) -> dict | None:
    """What the soil survey says is actually under this taluka.

    The feature store keeps only the numeric derivations of this — depth in
    millimetres, drainage as an ordinal, a black-soil flag — because that is
    what the gate consumes. The words themselves never reach a caller, and a
    caller now wants them: the app classifies a photograph of the farmer's own
    soil and has nothing to check the answer against.

    This was deliberately description rather than signal for as long as the
    engine took no image. It is now also the **prior** for `soil_fusion`, which
    weighs the farmer's photograph against it — see `recommend`.

    The words still never reach the gate directly. What reaches the gate is the
    fused soil type's effect on the soil-class flags and available water, and
    never on depth, drainage or salinity: those keep the surveyed value, which
    is what stops a misread photograph from lifting a veto.
    """
    from src.data.load import load_soil_type

    df = load_soil_type()
    hit = df[(df["District"] == district.upper()) & (df["Taluka"] == taluka.upper())]
    if hit.empty:
        return None
    r = hit.iloc[0]

    def _s(col: str):
        v = r.get(col)
        return None if v is None or pd.isna(v) else str(v)

    return {
        "soil_type": _s("Soil_Type"),
        "soil_type_secondary": _s("Soil_Type_Secondary"),
        "share_pct": None if pd.isna(r.get("Soil_Type_Share_pct"))
        else round(float(r["Soil_Type_Share_pct"]), 1),
        "texture": _s("Soil_Texture"),
        "depth": _s("Soil_Depth"),
        "drainage": _s("Soil_Drainage"),
        "parent_material": _s("Soil_Parent_Material"),
        "points_sampled": None if pd.isna(r.get("Soil_Points_Sampled"))
        else int(r["Soil_Points_Sampled"]),
    }


def _context(feats: dict, season: str, test: SoilTest,
             sources: dict[str, str] | None = None,
             fusion: soil_fusion.Fusion | None = None) -> dict:
    return {
        # The taluka as the soil survey records it. This was description only
        # for as long as the engine took no image — a photograph had nothing to
        # do but sit beside it in the UI. `recommend` now weighs the farmer's
        # photograph against it, so `soil_fusion` below says what the photograph
        # was allowed to do and whether it changed the answer at all.
        "surveyed_soil": _surveyed_soil(feats["District"], feats["Taluka"]),
        "soil_fusion": fusion.as_dict() if fusion is not None else None,
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
        # Per-field provenance for the two SHC components that feed rules
        # rather than the fertility table — mirrors "soil_test_source" above,
        # one level more granular. The taluka value actually used is echoed
        # too so a farmer's own reading is visibly reflected, not just claimed.
        # Per reading, because a card is often read in part: which of the twelve
        # came from this farmer's card and which from the taluka.
        "soil_test_sources": dict(sources or {}),
        "ph_source": (sources or {}).get(
            "pH", "farmer soil health card" if test.ph is not None
            else "taluka soil survey pH class"),
        "ph_used": round(float(feats["ph_class_value"]), 2),
        "ec_source": (sources or {}).get(
            "EC", "farmer soil health card" if test.ec_status is not None
            else "taluka SHC distribution"),
        "ec_saline_pct_used": round(float(feats["ec_saline"]), 1),
        "shc_samples": int(feats["n_samples"]),
    }
