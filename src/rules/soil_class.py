"""S4 Layer 1 — soil test values to fertility class (plan §6.1).

The fertiliser table is indexed by a Low/Medium/High ``Soil_Class`` backed by
exactly three nutrient archetypes. Those archetypes are precisely the national
Soil Health Card band midpoints, verified against the raw table:

    N   200 / 400 / 700 kg/ha      P    6 / 17 / 40 kg/ha
    K    80 / 190 / 350 kg/ha      OC  0.3 / 0.6 / 1.0 %

which means mapping a farmer's real test values to a class is a documented
rule, not a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class SoilTest:
    """A farmer's measured Soil Health Card values.

    N/P/K/OC drive the fertility class below, exactly as the government table
    does. ``ph``, ``ec_status`` and ``micronutrients`` carry the other eight
    SHC components — they have no fertility-class role (the table itself is
    only indexed by N/P/K/OC), but they do have a real place downstream: pH
    and salinity are Liebig-gate inputs (see ``pipeline.recommend``, which
    substitutes them for the taluka average when present), and micronutrient
    status drives the L4 correction layer (``rules.fertiliser.micronutrient_plan``).
    """
    n_kg_ha: float | None = None
    p_kg_ha: float | None = None
    k_kg_ha: float | None = None
    oc_pct: float | None = None
    #: measured soil pH, same scale as the taluka's ph_class_value
    ph: float | None = None
    #: the card's own low/normal/high verdict for EC; "high" means saline
    ec_status: str | None = None
    #: short code (config.MICRONUTRIENTS) -> "low"/"normal"/"high" from the card
    micronutrients: dict[str, str] | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {"N": self.n_kg_ha, "P": self.p_kg_ha, "K": self.k_kg_ha, "OC": self.oc_pct}


def classify_nutrient(component: str, value: float) -> str:
    """Map one measured value to Low / Medium / High using the SHC bands."""
    band = config.SOIL_BANDS[component]
    if value < band["low_max"]:
        return "Low"
    if value > band["high_min"]:
        return "High"
    return "Medium"


def classify(test: SoilTest) -> str:
    """Overall fertility class from a soil test.

    The table carries one class per recommendation, so the four components are
    reduced to a single class by majority vote with N as the tie-breaker —
    nitrogen drives the dose more than any other component.
    """
    classes = {
        comp: classify_nutrient(comp, val)
        for comp, val in test.as_dict().items()
        if val is not None
    }
    if not classes:
        raise ValueError("soil test carries no values")

    counts = {c: list(classes.values()).count(c) for c in config.SOIL_CLASSES}
    best = max(counts.values())
    winners = [c for c in config.SOIL_CLASSES if counts[c] == best]
    if len(winners) == 1:
        return winners[0]
    return classes.get("N", winners[0]) if classes.get("N") in winners else winners[0]


def taluka_soil_test(features: dict) -> SoilTest:
    """Fall back to the taluka's SHC distribution when the farmer has no test.

    The Nutrient Index is on [1, 3]; it is mapped onto the archetype anchors so
    that NI 1 -> Low anchor, NI 2 -> Medium anchor, NI 3 -> High anchor, with
    linear interpolation in between. This gives the interpolation layer in §6.2
    a sensible value to work from even with no farmer-supplied test.
    """
    def from_ni(component: str, ni: float) -> float:
        anchors = config.SOIL_ARCHETYPES[component]
        lo, mid, hi = anchors["Low"], anchors["Medium"], anchors["High"]
        ni = min(max(float(ni), 1.0), 3.0)
        return lo + (mid - lo) * (ni - 1) if ni <= 2 else mid + (hi - mid) * (ni - 2)

    return SoilTest(
        n_kg_ha=from_ni("N", features["NI_N"]),
        p_kg_ha=from_ni("P", features["NI_P"]),
        k_kg_ha=from_ni("K", features["NI_K"]),
        oc_pct=from_ni("OC", features["NI_OC"]),
    )
