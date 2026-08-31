"""S4 Layer 5 — least-cost fertiliser combination (plan §6.5).

The table publishes two options per crop-soil pair, but many combinations of
the available products satisfy the same N-P2O5-K2O target at very different
prices. For a farmer, cost per hectare is often the deciding factor, so this
turns the lookup into an optimiser.

    minimise    sum(price_i * qty_i)
    subject to  sum(qty_i * analysis_i,n) >= target_n   for each nutrient
                qty_i >= 0
                qty_i = 0 for locally unavailable products
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from src.rules.fertiliser import PRODUCT_ANALYSIS, PRODUCT_PRICE_INR_PER_KG

#: nutrients the LP constrains. Sulphur is included so the SSP route can be
#: forced when the micronutrient layer flags a sulphur deficiency.
LP_NUTRIENTS = ["N", "P2O5", "K2O", "S"]


def optimise(
    target: dict[str, float],
    *,
    prices: dict[str, float] | None = None,
    available: list[str] | None = None,
    require_sulphur_kg: float = 0.0,
    over_supply_tolerance: float = 0.15,
) -> dict:
    """Cheapest product mix meeting ``target`` (kg/ha of N, P2O5, K2O).

    ``over_supply_tolerance`` caps how far each nutrient may overshoot, so the
    LP cannot dump a cheap product that happens to carry a wanted nutrient.
    """
    prices = {**PRODUCT_PRICE_INR_PER_KG, **(prices or {})}
    products = [p for p in PRODUCT_ANALYSIS if p in prices]
    if available is not None:
        products = [p for p in products if p in available]
    if not products:
        return {"feasible": False, "reason": "no products available"}

    need = {
        "N": float(target.get("N") or 0.0),
        "P2O5": float(target.get("P2O5") or 0.0),
        "K2O": float(target.get("K2O") or 0.0),
        "S": float(require_sulphur_kg),
    }

    c = np.array([prices[p] for p in products], dtype=float)
    # -A_ub x <= -b  encodes  A x >= b
    a_lower = -np.array(
        [[PRODUCT_ANALYSIS[p].get(n, 0.0) for p in products] for n in LP_NUTRIENTS],
        dtype=float,
    )
    b_lower = -np.array([need[n] for n in LP_NUTRIENTS], dtype=float)
    # upper bound: never overshoot a *requested* nutrient by more than the tolerance
    a_upper = -a_lower
    b_upper = np.array(
        [need[n] * (1 + over_supply_tolerance) if need[n] > 0 else np.inf
         for n in LP_NUTRIENTS],
        dtype=float,
    )
    keep = np.isfinite(b_upper)

    res = linprog(
        c,
        A_ub=np.vstack([a_lower, a_upper[keep]]),
        b_ub=np.concatenate([b_lower, b_upper[keep]]),
        bounds=[(0, None)] * len(products),
        method="highs",
    )
    if not res.success:
        # relax the overshoot cap before giving up — a strict cap can make an
        # otherwise sensible target infeasible with only four products
        if over_supply_tolerance < 1.0:
            return optimise(target, prices=prices, available=available,
                            require_sulphur_kg=require_sulphur_kg,
                            over_supply_tolerance=over_supply_tolerance + 0.35)
        return {"feasible": False, "reason": res.message}

    qty = {p: round(float(q), 1) for p, q in zip(products, res.x) if q > 0.05}
    supplied = {
        n: round(sum(q * PRODUCT_ANALYSIS[p].get(n, 0.0) for p, q in qty.items()), 1)
        for n in LP_NUTRIENTS
    }
    return {
        "feasible": True,
        "products_kg_ha": qty,
        "cost_inr_per_ha": round(float(res.fun), 2),
        "nutrients_supplied": supplied,
        "target": {n: round(v, 1) for n, v in need.items() if v > 0},
    }


def cost_of(products: dict[str, float], prices: dict[str, float] | None = None) -> float | None:
    """Cost of a published table option, for comparison against the LP."""
    prices = {**PRODUCT_PRICE_INR_PER_KG, **(prices or {})}
    if not all(p in prices for p in products):
        return None
    return round(sum(prices[p] * q for p, q in products.items()), 2)


def compare_with_table(
    table_options: dict[int, dict[str, float]],
    target: dict[str, float],
    **kw,
) -> dict:
    """Cheapest published option vs the LP optimum, with the saving."""
    priced = {opt: cost_of(prods) for opt, prods in table_options.items()}
    priced = {o: c for o, c in priced.items() if c is not None}
    lp = optimise(target, **kw)
    if not priced or not lp["feasible"]:
        return {"lp": lp, "table_costs": priced, "saving_inr_per_ha": None}

    best_opt = min(priced, key=priced.get)
    saving = round(priced[best_opt] - lp["cost_inr_per_ha"], 2)
    return {
        "lp": lp,
        "table_costs": priced,
        "cheapest_table_option": best_opt,
        "cheapest_table_cost": priced[best_opt],
        "saving_inr_per_ha": saving,
        "saving_pct": round(100 * saving / priced[best_opt], 1) if priced[best_opt] else None,
    }
