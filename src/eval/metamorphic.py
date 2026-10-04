"""Metamorphic relations — each farmer input must move the answer the right way.

A metamorphic check does not need to know the right answer for a field. It
checks how the answer must *change* when one input changes, which is exactly
what a farmer relies on: that their own card, their own water and their own
season are really being read, and read in the right direction.

    M1  the card's pH moves away from a crop's optimum
            -> that crop's rule and learned scores never rise, and it is never un-vetoed
    M2  the card reads saline (EC high)
            -> the same, for every crop whose salinity tolerance is exceeded
    M3  rainfed -> irrigated
            -> nothing is newly vetoed: vetoes(irrigated) ⊆ vetoes(rainfed)
    M4  a season / water regime the fertiliser table publishes
            -> the recipe served is the one for that season and water regime
    M5  a soil photograph, at every class and maximum confidence
            -> never lifts a veto forced by depth, drainage or salinity
    M6  a card reading low N, low S, low Zn
            -> the N dose rises; the S and Zn corrections appear

``input_effects`` asks the complementary question: does each of the fifteen
inputs (twelve card readings, the photo, irrigation, season) change the
answer at all?
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

#: large enough for recommend() to list every candidate crop
ALL_CROPS = 60


@dataclass
class Check:
    test: str
    district: str
    taluka: str
    season: str
    crop: str | None
    passed: bool | None          # None: not applicable
    detail: str = ""


def _recommend(pipe, district, taluka, season, **kw):
    from src.pipeline import recommend
    return recommend(district, taluka, season, pipe=pipe, **kw)


def _scores(rec) -> dict:
    """crop -> (rule score, learned score, vetoed)."""
    out = {c.crop: (c.rule_score, c.learned_score, False) for c in rec.crops}
    for v in rec.vetoed:
        out[v["crop"]] = (v.get("score"), None, True)
    return out


def _envelope(apy_crop: str):
    from src.ontology.crop_map import to_fertiliser_crop
    from src.rules.crop_requirements import get
    fert = to_fertiliser_crop(apy_crop)
    return get(fert) if fert else None


def _no_rise(test, base, new, factor_worse, district, taluka, season) -> list[Check]:
    """Crops whose own factor got worse must not score higher, nor escape a veto."""
    b, n = _scores(base), _scores(new)
    checks = []
    for crop in b:
        env = _envelope(crop)
        if env is None or crop not in n or not factor_worse(env):
            continue
        rb, lb, vb = b[crop]
        rn, ln, vn = n[crop]
        bad = []
        if vb and not vn:
            bad.append("un-vetoed by a worse reading")
        if rb is not None and rn is not None and np.isfinite(rb) and np.isfinite(rn) \
                and rn > rb + 1e-9:
            bad.append(f"rule {rb:.3f} -> {rn:.3f}")
        if lb is not None and ln is not None and ln > lb + 1e-6:
            bad.append(f"learned {lb:.3f} -> {ln:.3f}")
        checks.append(Check(test, district, taluka, season, crop, not bad, "; ".join(bad)))
    return checks


def m1_ph(pipe, district, taluka, season, base) -> list[Check]:
    from src.pipeline import taluka_row
    from src.rules.soil_class import SoilTest
    from src.rules.suitability import trapezoid

    p0 = float(taluka_row(district, taluka)["ph_class_value"])
    new = _recommend(pipe, district, taluka, season,
                     soil_test=SoilTest(ph=5.0), top_k=ALL_CROPS)
    return _no_rise("M1", base, new,
                    lambda env: trapezoid(5.0, env.ph) < trapezoid(p0, env.ph) - 1e-9,
                    district, taluka, season)


def m2_ec(pipe, district, taluka, season, base) -> list[Check]:
    from src.pipeline import taluka_row
    from src.rules.soil_class import SoilTest
    from src.rules.suitability import _salinity_score

    s0 = float(taluka_row(district, taluka).get("ec_saline", 0.0))
    new = _recommend(pipe, district, taluka, season,
                     soil_test=SoilTest(ec_status="high"), top_k=ALL_CROPS)
    return _no_rise("M2", base, new,
                    lambda env: _salinity_score(100.0, env.max_saline_pct)
                    < _salinity_score(s0, env.max_saline_pct) - 1e-9,
                    district, taluka, season)


def m3_irrigation(pipe, district, taluka, season, base):
    irr = _recommend(pipe, district, taluka, season, irrigated=True, top_k=ALL_CROPS)
    extra = {v["crop"] for v in irr.vetoed} - {v["crop"] for v in base.vetoed}
    detail = ("newly vetoed under irrigation: " + ", ".join(sorted(extra))) if extra else ""
    return [Check("M3", district, taluka, season, None, not extra, detail)], irr


def _norm(v) -> str | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else str(v)


def _same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(abs(float(a[k]) - float(b[k])) < 1e-6 for k in a)


def _contexts(district: str, fert_crop: str, soil_class: str, option: int = 1) -> list:
    """Every (variety, irrigation, season) recipe the table publishes for this cell.

    The season and water regime are read with the engine's own tagger, so the
    check and the thing it checks cannot drift apart over a vocabulary
    disagreement ("Pre-kharif", say).
    """
    from src.rules import fertiliser as fz

    t = fz._table()
    sub = t[(t["District"] == district.upper())
            & (t["Crop"].str.casefold() == fert_crop.casefold())
            & (t["Soil_Class"] == soil_class) & (t["Option"] == option)]
    out = []
    for (var, irr, sea), g in sub.groupby(["Crop_Variety", "Crop_Irrigation", "Crop_Season"],
                                          dropna=False):
        out.append((fz._context_tags(irr, sea),
                    dict(zip(g["Fertilizer"], g["Quantity"].astype(float)))))
    return out


def _request_matches(tags: tuple, season: str, irrigated: bool) -> bool:
    """Does a published context satisfy the farmer's season and water regime?

    Only the dimensions a context actually states are checked: the table's two
    context columns are used inconsistently, and many recipes name neither.
    """
    from src.rules import fertiliser as fz

    stated_season, stated_water = tags
    want_season = fz.SEASON_TAGS.get((season or "").strip().casefold(), season)
    season_ok = pd.isna(stated_season) or stated_season is None or stated_season == want_season
    water_ok = pd.isna(stated_water) or stated_water is None or \
        stated_water == ("irrigated" if irrigated else "rainfed")
    return bool(season_ok and water_ok)


def m4_fertiliser(rec) -> list[Check]:
    """Where the table publishes the requested context, the served recipe must be it."""
    from src.ontology.crop_map import to_fertiliser_crop

    checks = []
    for c in rec.crops:
        f = c.fertiliser
        if not f or not f.get("available") or f.get("estimated"):
            continue
        fert = to_fertiliser_crop(c.crop)
        contexts = _contexts(rec.district, fert, f["soil_class"])
        if len(contexts) < 2:
            continue                       # no choice to get wrong
        matching = [p for tags, p in contexts if _request_matches(tags, rec.season, rec.irrigated)]
        if not matching or len(matching) == len(contexts):
            continue                       # the request does not discriminate
        served = f.get("table_products_kg_ha") or {}
        ok = any(_same(served, p) for p in matching)
        water = "irrigated" if rec.irrigated else "rainfed"
        checks.append(Check("M4", rec.district, rec.taluka, rec.season, c.crop, ok,
                            "" if ok else f"{water} {rec.season} request served a recipe "
                                          f"published for another context: {served}"))
    return checks


def m6_card(pipe, district, taluka, season) -> list[Check]:
    from src.rules.soil_class import SoilTest

    mid = dict(p_kg_ha=17.0, k_kg_ha=190.0, oc_pct=0.6)
    lo = _recommend(pipe, district, taluka, season, top_k=8,
                    soil_test=SoilTest(n_kg_ha=150.0, **mid))
    hi = _recommend(pipe, district, taluka, season, top_k=8,
                    soil_test=SoilTest(n_kg_ha=650.0, **mid))
    micro = _recommend(pipe, district, taluka, season, top_k=3,
                       soil_test=SoilTest(n_kg_ha=400.0, **mid,
                                          micronutrients={"S": "low", "Zn": "low"}))

    def n_target(rec):
        out = {}
        for c in rec.crops:
            f = c.fertiliser or {}
            if f.get("available"):
                n = (f.get("interpolated_target_kg_ha") or {}).get("N")
                if n is not None:
                    out[c.crop] = float(n)
        return out

    nl, nh = n_target(lo), n_target(hi)
    checks = [Check("M6", district, taluka, season, crop, nl[crop] >= nh[crop] - 1e-6,
                    f"N target at low-N card {nl[crop]:.1f} vs high-N card {nh[crop]:.1f}")
              for crop in nl if crop in nh]
    comps = {m["component"] for m in micro.micronutrients}
    for comp in ("S", "Zn"):
        checks.append(Check("M6", district, taluka, season, None, comp in comps,
                            f"{comp} correction {'present' if comp in comps else 'MISSING'}"))
    return checks


def sample_talukas(n_districts: int | None = None) -> list[tuple[str, str]]:
    """One taluka per district (the first, alphabetically) — every district covered."""
    from src.pipeline import _store
    s = _store().sort_values(["District", "Taluka"])
    firsts = s.groupby("District", sort=True).head(1)[["District", "Taluka"]]
    pairs = list(firsts.itertuples(index=False, name=None))
    if n_districts and n_districts < len(pairs):
        step = len(pairs) / n_districts
        pairs = [pairs[int(i * step)] for i in range(n_districts)]
    return pairs


def _confident_photos() -> dict[str, dict[str, float]]:
    """One maximally persuasive photograph per class the classifier knows.

    M5 asks what the *most* convincing possible photograph is allowed to do,
    not what an average one does, so each of these puts 0.99 on a single class.
    The class names are read from the deployed classifier's own metadata,
    because two different ones can be installed here and they spell their
    classes differently.
    """
    import json

    from src import config

    try:
        classes = list(json.loads(config.F_SOIL_MODEL_META.read_text())["classes"])
    except Exception:                                 # noqa: BLE001 - no classifier yet
        return {}
    rest = 0.01 / max(len(classes) - 1, 1)
    return {c: {k: (0.99 if k == c else rest) for k in classes} for c in classes}


def m5_photo(pipe, district, taluka, season, base) -> list["Check"]:
    """A photograph may add a constraint and may never remove one.

    ``soil_fusion.apply_to_features`` moves the soil-type flags and available
    water and never touches depth, drainage or salinity. So the crops vetoed on
    a **hard** factor must come out identical whatever the photograph claims.
    That is the relation which makes it safe to let a photograph into the
    answer at all: a wrong one can cost a farmer a recommendation, never their
    safety.
    """
    from src.rules.suitability import HARD_FACTORS

    def hard_vetoes(rec) -> set:
        return {v["crop"] for v in rec.vetoed
                if v.get("limiting_factor") in HARD_FACTORS}

    photos = _confident_photos()
    if not photos:
        return [Check("M5", district, taluka, season, None, None,
                      "no classifier is installed, so no photograph can be tested")]

    before = hard_vetoes(base)
    checks = []
    for label, probabilities in photos.items():
        rec = _recommend(pipe, district, taluka, season, top_k=ALL_CROPS,
                         soil_photo=probabilities)
        lifted = before - hard_vetoes(rec)
        checks.append(Check(
            "M5", district, taluka, season, label, not lifted,
            "" if not lifted else
            f"a {label} photograph lifted the hard-factor veto on "
            f"{', '.join(sorted(lifted))}"))
    return checks


def run(pipe, talukas, seasons=("Kharif", "Rabi")):
    """Every relation over a sample. Returns (checks frame, the recommendations made)."""
    checks: list[Check] = []
    recs = []
    for district, taluka in talukas:
        for season in seasons:
            base = _recommend(pipe, district, taluka, season, top_k=ALL_CROPS)
            checks += m1_ph(pipe, district, taluka, season, base)
            checks += m2_ec(pipe, district, taluka, season, base)
            m3, irr = m3_irrigation(pipe, district, taluka, season, base)
            checks += m3
            checks += m4_fertiliser(base) + m4_fertiliser(irr)
            checks += m5_photo(pipe, district, taluka, season, base)
            checks += m6_card(pipe, district, taluka, season)
            recs += [base, irr]
    return pd.DataFrame([asdict(c) for c in checks]), recs


# ------------------------------------------------------------ input effects --
def _signature(rec) -> tuple:
    crops = tuple(c.crop for c in rec.crops)
    vetoed = tuple(sorted(v["crop"] for v in rec.vetoed))
    fert = tuple(
        (c.crop,
         tuple(sorted(((c.fertiliser or {}).get("interpolated_target_kg_ha") or {}).items())),
         tuple(sorted(((c.fertiliser or {}).get("table_products_kg_ha") or {}).items())))
        for c in rec.crops)
    micro = tuple(sorted((m["component"], m.get("rate_kg_ha")) for m in rec.micronutrients))
    rules = tuple((c.crop, c.rule_score, c.learned_score) for c in rec.crops)
    # What the farmer is told about their own salinity. A card's EC "high" no
    # longer vetoes on its own (it cannot place the field in the survey's saline
    # class — see the EC note in `pipeline.recommend`), so it reaches the answer
    # as this caution and the crops it names; that is its effect, and it counts.
    ctx = rec.context or {}
    caution = (bool(ctx.get("ec_card_high")), tuple(ctx.get("ec_least_tolerant") or ()))
    return crops, vetoed, fert, micro, rules, caution


def _card_pairs():
    from src.rules.soil_class import SoilTest

    mid = dict(n_kg_ha=400.0, p_kg_ha=17.0, k_kg_ha=190.0, oc_pct=0.6)
    pairs = {
        "N": (SoilTest(**{**mid, "n_kg_ha": 150.0}), SoilTest(**{**mid, "n_kg_ha": 650.0})),
        "P": (SoilTest(**{**mid, "p_kg_ha": 5.0}), SoilTest(**{**mid, "p_kg_ha": 40.0})),
        "K": (SoilTest(**{**mid, "k_kg_ha": 70.0}), SoilTest(**{**mid, "k_kg_ha": 350.0})),
        # OC only votes on the fertility class; it can decide a tie between the
        # other three, so that is where its effect is looked for
        "OC": (SoilTest(n_kg_ha=250.0, p_kg_ha=17.0, k_kg_ha=190.0, oc_pct=0.3),
               SoilTest(n_kg_ha=250.0, p_kg_ha=17.0, k_kg_ha=190.0, oc_pct=1.0)),
        "pH": (SoilTest(ph=5.0), SoilTest(ph=7.0)),
        "EC": (SoilTest(ec_status="high"), SoilTest(ec_status="normal")),
    }
    for comp in ("S", "Zn", "Fe", "Mn", "Cu", "B"):
        pairs[comp] = (SoilTest(micronutrients={comp: "low"}),
                       SoilTest(micronutrients={comp: "normal"}))
    return pairs


def input_effects(pipe, talukas, season: str = "Kharif") -> pd.DataFrame:
    """Does each of the fifteen inputs change the answer for at least one taluka?"""
    import inspect

    from src.pipeline import recommend

    rows = []
    for name, (a, b) in _card_pairs().items():
        changed = sum(
            _signature(_recommend(pipe, d, t, season, soil_test=a, top_k=10))
            != _signature(_recommend(pipe, d, t, season, soil_test=b, top_k=10))
            for d, t in talukas)
        rows.append({"input": name, "kind": "card", "talukas_changed": changed,
                     "of": len(talukas), "effective": changed > 0})

    changed = sum(_signature(_recommend(pipe, d, t, "Rabi", irrigated=False, top_k=10))
                  != _signature(_recommend(pipe, d, t, "Rabi", irrigated=True, top_k=10))
                  for d, t in talukas)
    rows.append({"input": "irrigation", "kind": "farm", "talukas_changed": changed,
                 "of": len(talukas), "effective": changed > 0})

    changed = sum(_signature(_recommend(pipe, d, t, "Kharif", top_k=10))
                  != _signature(_recommend(pipe, d, t, "Rabi", top_k=10))
                  for d, t in talukas)
    rows.append({"input": "season", "kind": "farm", "talukas_changed": changed,
                 "of": len(talukas), "effective": changed > 0})

    # The photograph is measured the same way as every other input: does it
    # change the answer for at least one taluka? A zero here is a real finding,
    # not a gap — it means either that no classifier is installed or that the
    # survey prior is winning everywhere, and both deserve to be seen.
    if "soil_photo" not in inspect.signature(recommend).parameters:
        rows.append({"input": "soil photo", "kind": "photo", "talukas_changed": 0,
                     "of": len(talukas), "effective": False})
        return pd.DataFrame(rows)

    photos = list(_confident_photos().values())
    changed = 0
    for d, t in talukas:
        baseline = _signature(_recommend(pipe, d, t, season, top_k=10))
        if any(_signature(_recommend(pipe, d, t, season, top_k=10, soil_photo=p))
               != baseline for p in photos):
            changed += 1
    rows.append({"input": "soil photo", "kind": "photo", "talukas_changed": changed,
                 "of": len(talukas), "effective": changed > 0})
    return pd.DataFrame(rows)
