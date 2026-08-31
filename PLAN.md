# Implementation Plan — Crop & Fertiliser Recommendation Engine

Derived from `data/Crop_Fertiliser_ML_Training_Plan_1.md` (the project plan) and verified
against the five raw data files. Every structural claim in the plan was re-checked against
the actual data before this plan was written — results in "Data recon" below.

## Data recon (verified, not assumed)

| Plan claim | Verified |
|:--|:--|
| 351 SHC talukas, 358 soil-type talukas | ✅ exact |
| 7 urban talukas missing from SHC | ✅ Andheri, Borivali, Kurla, Nagpur (Urban), Pune City, Thane, Ulhasnagar |
| 6 duplicated taluka names | ✅ Ashti, Kalamb, Karanja, Karjat, Khed, Malegaon → **join on (District, Taluka)** |
| Districts reconcile across all 5 files | ✅ 34/34, no name drift after upper-casing |
| Weather: 358 talukas × 365 days = 130,670 | ✅ zero missing |
| APY: 1,000 rows, 34 districts, 25 crops, 4 seasons, **one year 2022-23** | ✅ |
| Fertiliser: 41,070 rows, 78 crops, 17 products | ✅ |
| `Data_Flag.isna()` → clean rows | ✅ 33,169 clean / 7,901 flagged (7,729 per-tree, 144 implausible, 18 incomplete, 40 option-diff) |
| Fertiliser table is a deterministic lookup | ✅ **0 ambiguous keys of 33,169** |
| Soil-class archetypes are band midpoints | ✅ N 200/400/700, P 6/17/40, K 80/190/350, OC 0.3/0.6/1.0 |

**One correction to the plan text:** §6.6 spells fertiliser crops as `Pearl millet`, `Pigeon pea`,
`Finger millet`, `Tetraploid cotton`. The table actually uses title case
(`Pearl Millet`, `Pigeon Pea`, `Finger Millet`, `Tetraploid Cotton`). The ontology therefore
resolves case-insensitively and is unit-tested against the real crop list.

---

## Phase 0 — Scaffold  *(plan §10)*
- `.venv` (Python 3.13) + pinned `requirements.txt`
- Repo tree exactly as §10: `data/{raw,interim,features}`, `src/{ontology,features,rules,models,eval,serve}`, `tests/`
- `src/config.py` — single source of paths, constants, season windows, RNG seed
- **Exit:** `pytest` collects; `make` targets run end-to-end

## Phase 1 — Data contracts & crop ontology  *(plan §9 week 1, §6.6)*
- `src/data/load.py` — canonicalise keys to `(DISTRICT, TALUKA)` upper-trimmed; BOM-safe reads
- Drop the 7 urban talukas with an explicit `urban_no_shc` flag (never impute)
- Filter fertiliser on `Data_Flag.isna()` at load
- `src/ontology/crop_map.py` — APY ↔ fertiliser names, 18/25 crops, 98.5% of area
- `src/data/quality_report.py` — emits the data-quality report (a thesis chapter)
- **Tests:** ontology covers 18 crops & every target exists in the table; duplicate-taluka join safety
- **Exit:** `data/interim/keys_canonical.parquet` + `reports/data_quality.md`

## Phase 2 — Feature store, Blocks A–D  *(plan §4)* → 351 × ~144
- **A · soil health (51)** — Nutrient Index (1–3), ILR on the 3-part L/M/H composition with
  `(x+0.5)/101.5` closure, `Deficient_pct` only for the 6 micronutrients, composites
  (`macro_NI`, `npk_imbalance`, ratios, `micro_def_count`, `micro_def_worst`, `ph_stress`,
  `n_samples_log`), contextual z-scores **behind a flag** (they hurt at n=34, §7.1)
- **B · soil physical (13)** — `awc_mm_m`, `depth_mm`, `drainage_ord` 1–6, `soil_purity`,
  and `rootzone_awc = awc × depth / 1000`
- **C · agro-climatology (60)** — Kharif/Rabi/Summer split; rainfall totals, longest dry spell,
  monsoon onset, 5-wettest-day concentration, rainy/heavy days, CV; **Hargreaves ET₀** from
  FAO-56 extraterrestrial radiation + latitude; aridity `P/ET₀`, CWB `P−ET₀`, LGP `P > 0.5·ET₀`;
  GDD base 10, heat days >35/>40, cold days <10, DTR, fungal-pressure proxy (RH>80 & Tmax 25–32)
- **D · interactions (20)** — `water_supply`, `drought_vuln`, `leach_risk`, `p_availability`,
  `zn_lockout`, `salinity_x_drain`, `irrig_need`, `knn5_*`/`anom_*` **behind a flag**
- **Tests:** ET₀ against FAO-56 worked example; dry-spell on synthetic series; NI/ILR identities;
  ranges match the plan's stated spans (rain 630–3,346 mm; rootzone_awc 21–270 mm; LGP 73–166 d)
- **Exit:** `data/features/taluka_features.parquet`, one deterministic command

## Phase 3 — Deterministic engines  *(plan §5.2, §6)* — a working recommender with no model
- `src/rules/suitability.py` — FAO EcoCrop/ICAR envelopes (rain, temp, pH, depth, drainage,
  salinity, LGP, texture) → per-factor 0–1 → **Liebig minimum** → S1/S2/S3/N; hard veto on N,
  every veto logged with its limiting factor (the explanation layer)
- `src/rules/soil_class.py` — SHC bands → Low/Medium/High
- `src/rules/fertiliser.py` — L2 exact lookup → L3 interpolation between the 200/400/700
  archetypes → L4 micronutrient corrections (S/Zn/Fe/B/Mn/Cu) → L5 `leach_risk` split schedule
- `src/rules/cost_optimiser.py` — `scipy.optimize.linprog` least-cost blend over the 17 products
- **Tests:** *fertiliser exactness* — 1,000 sampled keys reproduce the table byte-for-byte
  before any adjustment layer (plan §10, "day one")
- **Exit:** end-to-end rule-only recommendation for any taluka + season

## Phase 4 — Evaluation harness *first*, then models  *(plan §7)*
Built before the models so the validation scheme is never retrofitted.
- `src/eval/splits.py` — `GroupKFold(district)`, leave-one-district-out (34 folds), spatial blocks
- `src/eval/metrics.py` — NDCG@k, P@k, Recall@k, within-crop Spearman, MAE, conformal coverage
- `src/eval/baselines.py` — random + **out-of-fold popularity prior** (the 0.644 bar)
- **Test:** *leakage* — no district in both train and test of any fold
- Then:
  - `src/models/ranker.py` — LambdaMART, relevance `qcut(area_share.rank(), 5)` **within district**,
    groups = (district, season); target NDCG@5 ≈ 0.724 > popularity 0.644
  - `src/models/yield_quantile.py` — LightGBM quantile α = .1/.5/.9 on **within-crop z-scored**
    yield, converted back to t/ha at serve time; nested in-fold importance selection → ~35 features;
    10-seed bagging
- **Exit:** `src/eval/ablation.py` reproduces the §7.1 / §7.2 / §7.3 / §7.4 tables into `reports/`

## Phase 5 — Blending, conformal, abstention  *(plan §5.4)*
- Per-crop α from cross-validated within-crop Spearman; α → 0 where ρ < 0.2 (cotton, gram)
- Split-conformal intervals on a district-grouped calibration fold; report empirical coverage
- Mahalanobis OOD guard → suppress learned score, serve S2 alone, labelled as such
- **Exit:** `final = α·rank_norm(S1) + (1−α)·rank_norm(S2)` with a coverage table

## Phase 6 — Serving & explanations  *(plan §9 week 7)*
- `src/serve/api.py` — FastAPI: taluka + season + optional farmer SHC values →
  ranked crops, p10/p50/p90 yield band, dose plan + split schedule, cost-optimised blend,
  plain-language reason (S2 limiting factor + SHAP from S1), confidence / abstention flag
- Marathi crop names from the fertiliser table's `Crop_Local_Name`
- `src/cli.py` — same pipeline without a server

## Phase 7 — Reporting
- `reports/` — data quality, ablation, ranking benchmark, per-crop failure table, calibration
- `README.md` — reproduce-from-raw in one command; honest statement of the R²=0.230 ceiling,
  the weather-year mismatch, and the n=34 constraint
- §8 data-upgrade path documented as future work

---

## Non-negotiables carried from the plan
1. Join on **(District, Taluka)** — never on Taluka alone.
2. **GroupKFold by district** everywhere — never a random split.
3. Train on **within-crop z-scored yield** — never raw yield (that is the fake R²=0.928).
4. Fertiliser is a **lookup, never a model** — 0 ambiguous keys means a model can only add error.
5. Filter `Data_Flag.isna()` before any per-hectare arithmetic.
6. Feature **selection inside each CV fold** — selecting globally then CV-ing is a leak.
7. Report the **popularity baseline** beside every ranking number.
8. Weather is a **climatology descriptor** only — it is the wrong year for these yields.
