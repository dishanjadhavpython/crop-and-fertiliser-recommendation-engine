# Crop & Fertiliser Recommendation Engine — Maharashtra

A five-stage recommender built on Soil Health Card data, taluka agro-climatology
and district crop statistics, implementing
[`data/Crop_Fertiliser_ML_Training_Plan_1.md`](data/Crop_Fertiliser_ML_Training_Plan_1.md).

Each stage uses the technique that fits it — two learned models, two
deterministic engines, one calibration layer — rather than forcing everything
into a single classifier.

| Stage | Component | Type | What it does |
|:--|:--|:--|:--|
| **S0** | Taluka feature store | Pipeline | 351 talukas × 305 features from 18 raw files |
| **S1** | Crop suitability ranker | Learned | LambdaMART ranks candidate crops for a taluka + season |
| **S2** | Agronomic gate & scorer | Rules | FAO land-suitability envelopes; vetoes unsafe recommendations |
| **S3** | Yield & confidence | Learned | Quantile regression → p10/p50/p90 band |
| **S4** | Fertiliser engine | Lookup | Exact table lookup + interpolation + micronutrients + LP |
| **S5** | Conformal calibration | Hybrid | Coverage guarantees; abstains when out of distribution |

---

## Quick start

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# rebuild the feature store from the raw files (one deterministic command)
./.venv/bin/python -m src.features.build_store

# a recommendation
./.venv/bin/python -m src.cli --district SOLAPUR --taluka SANGOLE --season Rabi

# the reports
./.venv/bin/python -m src.data.quality_report      # reports/data_quality.md
./.venv/bin/python -m src.eval.ablation            # reports/benchmark_results.md

# the console — open http://127.0.0.1:8000
./.venv/bin/uvicorn src.serve.api:app --reload     # /docs for the API schema

./.venv/bin/python -m pytest tests -q              # 178 tests
./.venv/bin/python -m src.eval.experiments         # reports/experiments.md
```

---

## The panel

Eight years of crop statistics replaced the single year the original plan was
sized for — the change that plan named as worth more than every modelling
decision combined.

| | Original | Now |
|:--|--:|--:|
| APY years | 1 | **8** (2015-16 → 2022-23) |
| District-year label units | 34 | **272** |
| Labelled crop rows | 1,000 | **7,035** |
| Ranking queries | 136 | **1,088** |
| Weather years | 1 | **3** |
| Soil Health Card cycles | 1 | **3** |

Administrative geography is handled explicitly in
[`src/data/admin_changes.py`](src/data/admin_changes.py) — Palghar's 2014 split
from Thane, the three 2023 district renames, and the Soil Health Card coverage
that grew 347 → 351 talukas without a single taluka being created. Full
write-up, including one delivered file that had to be rejected, in
[reports/multiyear_upgrade.md](reports/multiyear_upgrade.md).

## The console

`uvicorn src.serve.api:app` then open `http://127.0.0.1:8000`. Pick a taluka and
a season, optionally paste in your own Soil Health Card numbers, and the page
calls the same pipeline the CLI does.

The design takes its palette from Maharashtra's own ground rather than a
generic agri-green — Deccan basalt, black cotton soil, turmeric and laterite.
Its signature element is the **Liebig stave chart**: the S2 gate literally
computes `score = min(eight factors)`, and agronomy has illustrated that law
since 1840 with a barrel whose shortest stave sets the water level. The chart
draws those eight staves, marks the shortest, and colours it by severity, so
"why this crop scored what it did" is the picture rather than a caption.

Every query is a URL, so a recommendation can be forwarded:
`?district=SOLAPUR&taluka=SANGOLE&season=Rabi`

## Headline results

All under **GroupKFold by district**. Nothing here uses a random split.

### The number you must not report

| Target | R² |
|:--|--:|
| Raw yield (t/ha), soil + weather features | **0.933** |
| Raw yield (t/ha), **crop + season identity only** | 0.914 |
| Within-crop z-scored yield, soil + weather | **0.190** |

A model that sees only the crop name scores 0.914. The impressive 0.933 is
almost entirely the model learning that sugarcane yields 74 t/ha and sesamum
0.27 t/ha. The honest within-crop signal is 0.190 — real, well above zero, and
the true ceiling this data supports.

### Ranking — the headline result

On the eight-year panel, GroupKFold by district:

| Method | NDCG@5 | NDCG@3 | P@3 | Recall@5 |
|:--|--:|--:|--:|--:|
| Random ranking | 0.153 | 0.125 | 0.112 | 0.186 |
| LambdaMART, crop identity only | 0.412 | 0.340 | 0.245 | 0.548 |
| S2 agronomic rules only *(no labels at all)* | 0.553 | 0.526 | 0.377 | 0.592 |
| Yield-regression framing | 0.730 | 0.680 | 0.459 | 0.761 |
| **Popularity prior, out-of-fold ← the bar** | **0.740** | 0.712 | 0.528 | 0.804 |
| LambdaMART + engineered features | 0.864 | 0.855 | 0.610 | 0.865 |
| **Full engine (early fusion + blend) ← as served** | **0.883** | **0.873** | **0.622** | **0.879** |

The engine clears the popularity baseline by **+0.143**. See
[reports/experiments.md](reports/experiments.md) for how it got there — the
first version scored *below* the baseline because of a defect in how the blend
weights were derived.

### Feature engineering beats model tuning

| Feature set | R² | Δ |
|:--|--:|--:|
| Crop + season identity only | −0.021 | — |
| + Soil Health Card (12 components) | 0.139 | **+0.160** |
| + Soil physical | 0.194 | **+0.055** |
| + Agro-climatic (engineered) | **0.205** | +0.011 |
| + Agronomic interactions | 0.190 | −0.015 |

Feature engineering moved R² by +0.23. Of every architectural lever tried —
nested feature selection, stronger regularisation, DART, seed bagging — the
best gained **+0.010** (10-seed bagging, 0.190 → 0.200) and the rest hurt,
badly in the case of heavy regularisation (0.099). The signal is genuinely
non-linear and the sample is genuinely too small; neither is fixable by tuning.
The Soil Health Card block alone delivers 68% of the achievable signal.

### What moved the number

Every arm below ran under the identical GroupKFold protocol; significance is a
Wilcoxon signed-rank test over the 136 per-query NDCG@5 scores.

| Change | Δ NDCG@5 | p |
|:--|--:|--:|
| Fix α: derive blend weights from **ranking** skill, not yield skill | **+0.063** | 6.9 × 10⁻⁵ |
| Early fusion: feed S2's label-free factor scores into S1 | **+0.035** | 2.0 × 10⁻⁴ |
| Ranker capacity 300 → 800 trees *(within-protocol control)* | +0.012 | 0.002 |

The α fix was a genuine defect. Blend weights govern S1, which is a *ranker*,
but they were being derived from the *yield* model's per-crop skill. The yield
model cannot order cotton (ρ = −0.33) or gram (ρ = −0.28) — but the ranker
orders both well (ρ = 0.68, 0.74). Ten crops were being routed to the weaker
rule scorer for no reason. After the fix, two crops fall back to rules, and
they are the right ones.

Honest negatives, none adopted: within-district dispersion features (p = 0.43),
taluka-level suitability fractions (p = 0.68), and four alternative relevance
labels — the plan's `qcut(area_share.rank(), 5)` was already optimal, and
binary relevance is significantly worse (−0.095).

### Eight years vs one

| Panel | Protocol | Popularity prior | Engine | Margin |
|:--|:--|--:|--:|--:|
| 1 year | GroupKFold | 0.790 | 0.873 | +0.083 |
| **8 years** | GroupKFold | 0.740 | 0.862 | **+0.122** |
| **8 years** | forward chaining *(train past → rank next season)* | 0.740 | **0.888** | **+0.148** |
| 8 years | grouped + temporal *(unseen district AND year)* | 0.757 | 0.805 | +0.048 |

Absolute NDCG@5 dips slightly under GroupKFold because the eight-year task is
harder — 28 crops instead of 25, across a decade of shifting patterns. **The
margin over the baseline is what matters, and it widened from +0.083 to
+0.122.** Recall@5 improved 0.841 → 0.878.

Forward chaining is the deployment number: train on all history, rank next
season. Grouped + temporal is the strictest honest claim — a fold shares
neither a district nor a year with its training set, and the engine still
clears the baseline.

### Yield: the two crops the plan called broken are fixed

| Crop | ρ, 1 year | ρ, 8 years |
|:--|--:|--:|
| Cotton (lint) | **−0.33** | **+0.375** |
| Gram | **−0.28** | **+0.285** |

Within-crop ρ rose **0.276 → 0.426**. Crops where the model is worse than
useless (ρ < 0.2) fell from **10 of 25 to 2 of 28** — without adding the
irrigation variable the plan blamed for cotton and gram.

S3 now also reports a **yield class** (below / typical / above that crop's own
tercile), which the data supports far better than a number: 50.8% accuracy
against a 34.2% majority baseline, with severe errors — calling "below" when
the truth is "above" — at 12%.

### Calibration

| Interval | Empirical coverage | Mean width |
|:--|--:|--:|
| Raw quantile model, nominal 80% | 0.704 | 1.80 |
| Split-conformal, nominal 80% | **0.803** | 2.11 |
| Split-conformal, nominal 90% | **0.901** | 2.70 |

A 90% interval covering 90.1% of held-out districts is a checkable claim, not
an aspiration.

---

### The leakage warning is now demonstrable

| Protocol | R² |
|:--|--:|
| **Random KFold — LEAKS, districts split across folds** | **0.325** |
| GroupKFold by district | 0.191 |
| Leave-one-district-out | 0.163 |
| Spatial block CV | 0.053 |

On a single year of labels this comparison was inconclusive — random splitting
scored *lower* than GroupKFold, because there was only one row per
district-crop-season to leak. With eight years there are siblings to leak, and
a random split inflates R² by **70%**. This is the plan's Rule One, finally
shown rather than asserted.

## What this system will not claim

These limits are in the code, the reports and the API output, not just here.

**The weather is the wrong year.** Weather covers Apr 2025 – Mar 2026; the
yields are 2022-23. Weather is used strictly as a *climatology descriptor* —
"what this place is typically like" — never as a causal signal. With a single
year there is no inter-annual variation, so no model here can learn how a crop
responds to a dry year versus a wet one.

**The effective sample size is 34, not 1,000.** Features are at taluka grain
(351 rows); labels are at district grain (34 districts). Every modelling
decision is sized for 34. The ablation shows the constraint binding: past ~147
features, accuracy falls.

**Area share is not pure agronomy.** The ranker learns revealed farmer
preference, which is confounded by irrigation access, MSP policy and sugar
co-operative politics. Sugarcane's dominance in Kolhapur is partly
institutional.

**The model fails on some crops, and says so.** Within-crop Spearman runs from
0.76 (sesamum) to −0.33 (cotton) and −0.28 (gram). Both are irrigation- and
market-determined, and there is no irrigation variable in this data. Where
ρ ≤ 0.2 the blend weight α goes to zero and the rule scorer carries the
recommendation entirely — 10 of 25 crops are decided by rules alone.

**Suitability is scored rainfed by default.** 263 of 351 talukas support a
rainfed Rabi crop; irrigation lifts that to 341. The 78-taluka gap is reported,
not hidden, and `--irrigated` makes the assumption explicit.

---

## Findings not in the plan

Recorded in full in [`reports/data_quality.md`](reports/data_quality.md).

1. **The fertiliser table is incomplete across districts.** Only 63.7% of
   (mapped crop × district) cells carry a published recipe — linseed appears in
   3 districts of 34. Doses genuinely vary by district, so borrowing a
   neighbour's recipe would be wrong; the engine serves a state-wide median
   instead, labelled as an estimate and carrying its own observed spread.

2. **The least-cost LP does not save money.** With only four straight
   fertilisers the government's Option 1 is already cost-optimal — across 400
   sampled keys the LP never beat it by more than ₹0.23/ha. It earns its place
   only where the table has no answer at all: a locally unavailable product, or
   sulphur the micronutrient layer requires. An honest negative result.

3. **Rabi cannot be scored on in-season rainfall.** Rabi rain averages ~60 mm
   state-wide; Maharashtra's Rabi crops run on monsoon moisture stored in the
   profile. The gate adds the root-zone store to the season's rain — without it,
   every Rabi crop in the state is vetoed.

4. **Feature *ordering* moves R² by ±0.016** — the same magnitude as every
   architectural lever in §7.2 — because LightGBM's column subsampling depends
   on column order. The ablation fixes a canonical order so its rows are
   comparable.

5. **The plan's crop spellings and texture vocabulary don't match the data.**
   §6.6 writes `Pearl millet`; the table says `Pearl Millet`. §4.2 assumes a
   Sandy→Clayey texture scale; the data has only Clayey, Loamy and two
   `-skeletal` variants. Both are handled and unit-tested.

---

## Repository layout

```
data/raw/                        the 5 source files, never modified
data/features/taluka_features.parquet    351 x 161, one command from raw

src/config.py                    paths, constants, thresholds — one source of truth
src/data/load.py                 loaders enforcing the (District, Taluka) contract
src/data/training_set.py         district-grain modelling frame, both targets
src/data/quality_report.py       -> reports/data_quality.md
src/ontology/crop_map.py         APY <-> fertiliser names          §6.6

# deterministic — no training, fully testable
src/features/soil_health.py      Block A: nutrient index, ILR, composites
src/features/soil_physical.py    Block B: AWC, root-zone water
src/features/agroclimate.py      Block C: Hargreaves ET0, LGP, dry spells
src/features/interactions.py     Block D: leach risk, drought vulnerability
src/features/agronomic_fit.py    early fusion of S2 into S1 — label-free
src/features/climatology.py      3-year normals, inter-annual variability, SHC trend
src/data/admin_changes.py        renames, splits, coverage onset — the panel's geography
src/rules/crop_requirements.py   26 FAO EcoCrop / ICAR envelopes
src/rules/suitability.py         S2: Liebig minimum, hard vetoes
src/rules/soil_class.py          S4 L1: SHC band thresholds
src/rules/fertiliser.py          S4 L2-5: lookup, interpolate, micronutrients, splits
src/rules/cost_optimiser.py      S4 L6: scipy linprog

# learned
src/models/ranker.py             S1: LGBMRanker, lambdarank
src/models/yield_quantile.py     S3: alpha = .1/.5/.9, nested in-fold selection
src/models/conformal.py          S5: split-conformal + Mahalanobis OOD guard
src/models/yield_class.py        S3b: below / typical / above tercile
src/models/blend.py              S5: per-crop alpha weights

# evaluation
src/eval/splits.py               GroupKFold(district), LODO, spatial blocks,
                                 forward chaining, grouped+temporal
src/eval/metrics.py              NDCG@k, P@k, within-crop rho, coverage
src/eval/baselines.py            random, popularity prior, rules-only
src/eval/ablation.py             -> reports/benchmark_results.md
src/eval/experiments.py          -> reports/experiments.md  (improvement arms)

src/pipeline.py                  the five stages wired together
src/serve/api.py                 FastAPI + serves the console at /
src/serve/static/                index.html · app.css · app.js  (the console)
src/cli.py                       same pipeline, no server
tests/                           178 tests
```

## The tests that matter

Both of the plan's "write these on day one" tests are in CI:

- **`tests/test_no_leakage.py`** — no district appears in both train and test of
  any fold, in any splitter; relevance is graded within district; the popularity
  prior is out-of-fold; feature selection happens inside the fold.
- **`tests/test_fertiliser_exactness.py`** — 1,000 sampled keys reproduce the
  government table exactly before any adjustment layer, and the table is
  re-verified as a deterministic function (0 ambiguous keys of 33,169).

Plus `tests/test_suitability.py` asserting the gate is regionally correct —
Kolhapur ranks sugarcane first, semi-arid Solapur ranks chickpea and safflower,
and rice is vetoed on excessively-drained uplands.

---

## Next: the data upgrade path

The benchmark makes the ceiling clear — model changes moved R² by ±0.03,
feature engineering by +0.23, and the missing data is worth more than both.

1. **Multi-year APY** (data.gov.in, 1997→2023) — turns 34 training samples into
   ~800. The single highest-value addition; every weak result above is a
   symptom of n=34.
2. **Multi-year daily weather** (IMD Pune) — removes the year-mismatch entirely
   and allows true 30-year climatology.
3. **Irrigation coverage** by taluka — the biggest missing confounder; cotton
   and gram sit at negative ρ and both are irrigation-determined.
4. **Market prices** (Agmarknet) — lets the ranker optimise gross margin per
   hectare instead of yield. Farmers optimise rupees, not tonnes.
