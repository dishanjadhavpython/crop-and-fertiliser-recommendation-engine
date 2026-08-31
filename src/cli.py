"""Command-line access to the same pipeline the API serves.

    python -m src.cli --district KOLHAPUR --taluka KARVIR --season Kharif
    python -m src.cli --district SOLAPUR --taluka SANGOLE --season Rabi --irrigated
    python -m src.cli --district NASHIK --taluka NIPHAD --season Rabi --n 210 --p 8 --k 150 --oc 0.4
"""
from __future__ import annotations

import argparse
import warnings

from src import config
from src.pipeline import recommend
from src.rules.soil_class import SoilTest

warnings.filterwarnings("ignore")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--district", required=True)
    p.add_argument("--taluka", required=True)
    p.add_argument("--season", required=True, choices=config.APY_SEASONS)
    p.add_argument("--irrigated", action="store_true",
                   help="the farmer can supply the seasonal water deficit")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--n", type=float, help="available nitrogen, kg/ha")
    p.add_argument("--p", type=float, help="available phosphorus, kg/ha")
    p.add_argument("--k", type=float, help="available potassium, kg/ha")
    p.add_argument("--oc", type=float, help="organic carbon, %%")
    p.add_argument("--json", action="store_true", help="emit the full JSON payload")
    args = p.parse_args()

    test = None
    if any(v is not None for v in (args.n, args.p, args.k, args.oc)):
        test = SoilTest(args.n, args.p, args.k, args.oc)

    rec = recommend(args.district, args.taluka, args.season,
                    soil_test=test, irrigated=args.irrigated, top_k=args.top_k)

    if args.json:
        print(rec.to_json(indent=1))
        return
    _print(rec)


def _print(rec) -> None:
    bar = "=" * 78
    print(bar)
    print(f"{rec.taluka}, {rec.district} — {rec.season}"
          f"{'  (irrigated)' if rec.irrigated else '  (rainfed)'}")
    print(bar)
    c = rec.context
    print(f"  rainfall {c['annual_rainfall_mm']:.0f} mm/yr · aridity {c['aridity_index']} · "
          f"LGP {c['lgp_days']:.0f} d · root-zone water {c['rootzone_awc_mm']:.0f} mm")
    print(f"  soil fertility class: {rec.soil_class}  (from {rec.soil_test_source})")
    if not rec.confident:
        print(f"\n  ! {rec.abstention_reason}")
    if rec.water_limited:
        print("\n  ! Every crop is vetoed on water alone. This season needs irrigation here.")

    print(f"\n  RECOMMENDED CROPS")
    for a in rec.crops:
        name = f"{a.crop} ({a.crop_marathi})" if a.crop_marathi else a.crop
        print(f"\n  {a.rank}. {name}")
        print(f"     score {a.final_score:.3f} · suitability {a.suitability_class} · "
              f"decided by {a.decided_by}")
        if a.yield_class:
            conf = f" (confidence {a.yield_class_confidence:.0%})" if a.yield_class_confidence else ""
            regime = f" · {a.yield_regime}" if a.yield_regime else ""
            print(f"     yield  {a.yield_class.upper()} the norm for this crop{conf}{regime}")
        if a.yield_p50_t_ha is not None:
            print(f"            {a.yield_p10_t_ha}–{a.yield_p90_t_ha} t/ha "
                  f"(median {a.yield_p50_t_ha})")
        elif a.yield_abstained:
            print(f"            no range offered — insufficient skill for this crop")
        print(f"     why    {a.reason}")
        f = a.fertiliser
        if f and not f.get("available"):
            print(f"     dose   {f.get('note', 'no published recommendation')}")
        if f and f.get("available"):
            t = f["interpolated_target_kg_ha"]
            flag = "  [state-median estimate]" if f.get("estimated") else ""
            print(f"     dose   N {t['N']:.0f} · P2O5 {t['P2O5']:.0f} · K2O {t['K2O']:.0f} kg/ha "
                  f"(soil class {f['soil_class']}){flag}")
            mix = f["recommended_mix"]
            if mix.get("feasible"):
                blend = " + ".join(f"{k} {v:.0f} kg" for k, v in mix["products_kg_ha"].items())
                print(f"     mix    {blend}  ≈ ₹{mix['cost_inr_per_ha']:,.0f}/ha")
            sched = f["nitrogen_schedule"]
            splits = " → ".join(f"{s['n_kg_ha']:.0f} kg N at {s['stage']}"
                                for s in sched["schedule"])
            print(f"     split  {splits}   ({sched['leach_risk_band']} leaching risk)")
            if f.get("sulphur_swap"):
                print(f"     note   {f['sulphur_swap']['note']}")

    if rec.micronutrients:
        print(f"\n  MICRONUTRIENT CORRECTIONS  (the six components the government table ignores)")
        for m in rec.micronutrients:
            print(f"     {m['component']:<3} {m['deficient_pct']:>5.1f}% of samples deficient "
                  f"→ {m['product']} {m['rate_kg_ha']:.0f} kg/ha  [{m['priority']}]")
            print(f"         {m['note']}")

    if rec.not_assessable:
        print(f"\n  NOT ASSESSABLE ({len(rec.not_assessable)})  — no agronomic envelope "
              f"and no fertiliser recipe")
        print("     " + ", ".join(x["crop"] for x in rec.not_assessable))

    if rec.vetoed:
        print(f"\n  VETOED BY THE AGRONOMIC GATE ({len(rec.vetoed)})")
        for v in rec.vetoed[:8]:
            print(f"     {v['crop']:<22} {v['limiting_factor']:<10} {v['reason'][:70]}")
    print()


if __name__ == "__main__":
    main()
