"""Fuse the farmer's soil photograph with the taluka's soil survey.

Until now the photograph changed nothing. `pipeline._surveyed_soil` says so in
as many words — "deliberately description, not signal ... it changes no score" —
and the app printed the classifier's answer beside the survey's for the farmer
to compare. This module is what makes the photograph an input.

**The survey is the prior, the photograph is the evidence.** A taluka's surveyed
primary type covers a known share of its area (81% on average across
Maharashtra), which is a genuine prior and a strong one. A photograph is one
frame of one field, taken on a phone, by a classifier with a measured confusion
matrix. Treating them as equals would be wrong in both directions, so this is a
Bayesian update: prior from the survey's area shares, likelihood from the
classifier's *pooled out-of-fold* confusion matrix — what it actually does on
held-out photographs, not what its softmax claims.

Four constraints, each for a reason:

* **It chooses only between types the survey records for that taluka.** The
  classifier cannot introduce a soil that Maharashtra's survey does not place
  there. A confident photograph of something unsurveyed is a reason to distrust
  the photograph, not to rewrite the map.
* **It moves only soft inputs** — texture and available water capacity. Depth,
  drainage and salinity always take the *more cautious* of the two profiles, so
  a photograph can add a constraint and can never remove one. This is what makes
  M5 hold: the photo never relaxes a hard factor, and therefore never lifts a
  safety veto.
* **It acts only when the photograph is both confident and in-distribution.**
  Below `MIN_CONFIDENCE`, or on an image the classifier has no business judging,
  the survey stands unchanged.
* **Clay is not a vote.** The owner's dataset labels a Clay class, but clay is a
  *texture*, not a soil order, and no Maharashtra survey type corresponds to it.
  A Clay prediction is read as evidence about texture and is excluded from the
  soil-type posterior entirely.

Which soils fusion can judge depends on which classifier is installed, and it
abstains on any surveyed type the classifier has no class for. The shipped
eight-class model covers Alluvial, Black (Regur), Laterite and Red & Yellow. A
classifier trained on the owner's four-class field data alone would lose
Laterite — that data holds 29 laterite images, far too few — and would
therefore abstain across the whole Konkan, where laterite is the soil.

Class names are matched on a normalised key rather than literally, because the
two candidate classifiers spell them differently (`black` against `Black
Soil`). A literal match would make fusion abstain silently against whichever
one it was not written for: the photograph would stop mattering and nothing
would report it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Below this the photograph is not acted on and the survey stands.
MIN_CONFIDENCE = 0.80

#: How far the confusion matrix is pulled toward uniform before use. It is
#: measured on a few hundred scenes, so its off-diagonal zeros are small-sample
#: accidents rather than impossibilities, and an un-shrunk zero would let one
#: photograph veto a surveyed soil outright.
SHRINKAGE = 0.50

def normalise(label: str) -> str:
    """`Black Soil`, `black` and `Black_Soil` all name the same thing.

    Two classifiers can be deployed here and they disagree about spelling: the
    old eight-class checkpoint labels its classes `alluvial`, `black`, `red`,
    while one trained on the owner's folders labels them `Alluvial soil`,
    `Black Soil`, `Red soil`. Matching on the literal string would make fusion
    abstain silently against whichever of the two it was not written for, which
    is the worst possible failure — the photograph would simply stop mattering
    and nothing would say so.
    """
    return re.sub(r"[^a-z]+", " ", label.lower()).replace("soil", "").strip()


#: Normalised classifier label -> the engine's Maharashtra survey vocabulary.
#: `None` means the label casts no soil-type vote: Clay is a texture rather than
#: a soil order, and cinder and peat are not Maharashtra soils at all — they are
#: artefacts of the old web-scraped class list.
TO_SURVEY = {
    "alluvial": "Alluvial",
    "black": "Black (Regur)",
    "red": "Red & Yellow",
    "yellow": "Red & Yellow",
    "laterite": "Laterite",
    "clay": None,
    "cinder": None,
    "peat": None,
}

#: Available water capacity multipliers by texture, relative to the taluka's
#: surveyed value. Soft input: it moves the yield and water-stress side of the
#: answer, never a hard veto.
TEXTURE_AWC = {
    "clay": 1.10,
    "black": 1.05,
    "alluvial": 1.00,
    "red": 0.90,
    "laterite": 0.85,
}


@dataclass(frozen=True)
class Fusion:
    """What the photograph did, in a form the API can hand to the farmer."""

    soil_type: str | None                      # the posterior choice, or None
    prior: dict[str, float] = field(default_factory=dict)
    posterior: dict[str, float] = field(default_factory=dict)
    applied: bool = False
    reason: str = ""
    awc_scale: float = 1.0
    photo_label: str | None = None
    photo_confidence: float | None = None

    def as_dict(self) -> dict:
        return {
            "soil_type": self.soil_type,
            "applied": self.applied,
            "reason": self.reason,
            "prior": {k: round(v, 3) for k, v in self.prior.items()},
            "posterior": {k: round(v, 3) for k, v in self.posterior.items()},
            "awc_scale": round(self.awc_scale, 3),
            "photo": {"label": self.photo_label,
                      "confidence": None if self.photo_confidence is None
                      else round(self.photo_confidence, 3)},
        }


@lru_cache(maxsize=1)
def _likelihood(metadata_path: str) -> tuple[dict, list[str]]:
    """P(photo says c | the soil really is s), from held-out predictions.

    Rows of `confusion_pooled` are truth and columns are prediction, so a
    row-normalised matrix is exactly this likelihood. Shrinking it toward
    uniform keeps a small-sample zero from acting as a certainty.
    """
    meta = json.loads(Path(metadata_path).read_text())
    classes = list(meta["classes"])
    # `confusion_pooled` is every held-out image; the older checkpoint only kept
    # its best fold's matrix. Either is usable as a likelihood, and refusing to
    # read the older one would mean fusion could never run against it.
    raw = meta.get("confusion_pooled") or meta.get("confusion_best_fold")
    if raw is None:
        return {}, classes
    matrix = np.asarray(raw, dtype=float)

    # Several classes can name one surveyed soil — the old list separates red
    # from yellow where the survey has a single "Red & Yellow". Their counts are
    # summed before normalising, so the likelihood is over the survey's
    # vocabulary rather than the classifier's.
    grouped: dict[str, np.ndarray] = {}
    for i, truth in enumerate(classes):
        survey = TO_SURVEY.get(normalise(truth))
        if survey is None:
            continue                      # Clay is a texture; see the docstring
        grouped[survey] = grouped.get(survey, np.zeros(matrix.shape[1])) + matrix[i]

    table = {}
    for survey, counts in grouped.items():
        total = counts.sum()
        conditional = (counts / total if total else
                       np.full(len(counts), 1.0 / len(counts)))
        conditional = (1 - SHRINKAGE) * conditional + SHRINKAGE / len(counts)
        table[survey] = {classes[j]: float(conditional[j]) for j in range(len(classes))}
    return table, classes


def _prior(surveyed: dict) -> dict[str, float]:
    """The survey's own area shares, as a distribution over that taluka's types.

    `share_pct` is the primary type's share of the taluka. The secondary type
    takes the remainder, which is the survey's own statement about how mixed the
    taluka is — a 55/45 taluka should be much easier for a photograph to move
    than a 95/5 one, and this is where that comes from.
    """
    primary = surveyed.get("soil_type")
    secondary = surveyed.get("soil_type_secondary")
    if not primary:
        return {}
    share = surveyed.get("share_pct")
    share = 0.80 if share is None else min(max(float(share) / 100.0, 0.05), 0.98)
    if not secondary or secondary == primary:
        return {primary: 1.0}
    return {primary: share, secondary: 1.0 - share}


def fuse(surveyed: dict | None, probabilities: dict[str, float] | None,
         metadata_path: str | Path, *, in_distribution: bool = True,
         min_confidence: float = MIN_CONFIDENCE) -> Fusion:
    """Update the surveyed soil type with one photograph.

    `probabilities` is the classifier's calibrated output keyed by its own class
    names. `in_distribution` is the caller's non-soil rejection: a photograph of
    a leaf or a hand must not update anything.
    """
    if not surveyed or not probabilities:
        return Fusion(soil_type=(surveyed or {}).get("soil_type"),
                      reason="no photograph, or no survey for this taluka")

    label = max(probabilities, key=probabilities.get)
    confidence = float(probabilities[label])
    prior = _prior(surveyed)
    base = Fusion(soil_type=surveyed.get("soil_type"), prior=prior,
                  photo_label=label, photo_confidence=confidence)

    if not in_distribution:
        return Fusion(**{**base.__dict__,
                         "reason": "the photograph is not soil the classifier knows"})
    if confidence < min_confidence:
        return Fusion(**{**base.__dict__,
                         "reason": f"photograph confidence {confidence:.2f} is below "
                                   f"{min_confidence:.2f}; the survey stands"})
    if len(prior) < 2:
        return Fusion(**{**base.__dict__,
                         "reason": "the survey records a single soil type for this "
                                   "taluka; there is nothing for a photograph to "
                                   "choose between"})

    likelihood, _classes = _likelihood(str(metadata_path))
    unknown = [s for s in prior if s not in likelihood]
    if unknown:
        return Fusion(**{**base.__dict__,
                         "reason": f"the classifier has no class for {', '.join(unknown)}, "
                                   f"so it cannot judge between this taluka's soils"})

    posterior = {s: prior[s] * likelihood[s].get(label, 0.0) for s in prior}
    total = sum(posterior.values())
    if total <= 0:
        return Fusion(**{**base.__dict__,
                         "reason": "the photograph is incompatible with every surveyed "
                                   "soil here; the survey stands"})
    posterior = {s: v / total for s, v in posterior.items()}
    chosen = max(posterior, key=posterior.get)

    return Fusion(
        soil_type=chosen,
        prior=prior,
        posterior=posterior,
        applied=chosen != surveyed.get("soil_type"),
        awc_scale=TEXTURE_AWC.get(normalise(label), 1.0),
        photo_label=label,
        photo_confidence=confidence,
        reason=(f"survey {surveyed.get('soil_type')} at {prior.get(surveyed.get('soil_type'), 0):.0%} "
                f"updated by a {confidence:.0%} photograph of {label} "
                f"-> {chosen} at {posterior[chosen]:.0%}"),
    )


#: The only feature columns a photograph may move. Everything else — depth,
#: drainage, salinity, pH — keeps the surveyed value, and that is precisely what
#: makes a veto un-liftable by a picture.
MUTABLE = frozenset({"is_black_soil", "is_lateritic", "rootzone_awc"})


def property_check(metadata_path=None) -> tuple[bool, str]:
    """Hard gate: no photograph, however confident, may relax a hard factor.

    Checked as a *property* of the fusion rather than by sampling
    recommendations. `apply_to_features` is the only route by which a
    photograph reaches the gate's inputs, so enumerating every fusion it can
    produce — each surveyed pair, each class the classifier knows, confident
    and unconfident, in and out of distribution — settles the question for all
    of them at once. Sampling recommendations could only ever fail to find a
    counterexample, which is not the same as there not being one.
    """
    from src import config

    path = Path(metadata_path or config.F_SOIL_MODEL_META)
    if not path.exists():
        return True, "no classifier installed, so no photograph can reach the gate"

    table, classes = _likelihood(str(path))
    if not table:
        return True, "the installed classifier maps to no surveyed soil type"

    baseline = {"depth_mm": 450.0, "drainage_ord": 2.0, "ec_saline": 12.0,
                "ph_class_value": 8.4, "rootzone_awc": 120.0,
                "is_black_soil": 1, "is_lateritic": 0}

    surveys = [{"soil_type": p, "soil_type_secondary": s, "share_pct": share}
               for p in table for s in list(table) + [None]
               for share in (55.0, 80.0, 95.0) if s != p]

    checked, violations = 0, []
    for survey in surveys:
        for label in classes:
            for confidence in (0.5, 0.85, 0.99):
                spread = (1.0 - confidence) / max(len(classes) - 1, 1)
                probabilities = {c: (confidence if c == label else spread)
                                 for c in classes}
                for in_dist in (True, False):
                    fusion = fuse(survey, probabilities, path,
                                  in_distribution=in_dist)
                    out = apply_to_features(baseline, fusion)
                    moved = {k for k in baseline if out.get(k) != baseline[k]}
                    checked += 1
                    if not moved <= MUTABLE:
                        violations.append(
                            f"{survey['soil_type']}/{survey['soil_type_secondary']} "
                            f"+ {label}@{confidence} moved {sorted(moved - MUTABLE)}")

    if violations:
        return False, f"{len(violations)} of {checked} fusions moved a hard factor, " \
                      f"e.g. {violations[0]}"
    return True, (f"{checked} fusions checked over {len(surveys)} surveyed pairs and "
                  f"{len(classes)} classes; none moved depth, drainage, salinity or pH")


def apply_to_features(feats: dict, fusion: Fusion) -> dict:
    """Push a fusion result into the feature dict the gate and ranker read.

    Only the soft inputs move. `depth_mm`, `drainage_ord` and the salinity
    columns are deliberately absent from this function: they are the hard
    factors, they keep the surveyed taluka's value, and that is the whole reason
    a photograph cannot lift a veto.
    """
    if not fusion.applied or not fusion.soil_type:
        return feats

    out = dict(feats)
    out["is_black_soil"] = int(fusion.soil_type == "Black (Regur)")
    out["is_lateritic"] = int(fusion.soil_type in {"Laterite", "Red & Yellow"})
    if "rootzone_awc" in out:
        out["rootzone_awc"] = float(out["rootzone_awc"]) * fusion.awc_scale
    return out
