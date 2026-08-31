# Improvement plan — S2 agronomic gate and S3 yield

Both stages scored 6/10 in the last review. This plan takes them to 8.5–9.
Every target below is grounded in a diagnostic already run against the eight-year
panel, not in an estimate — the measurements are quoted inline.

---

# Part 1 · S2 — the agronomic gate

## The diagnosis

The gate had never been checked against data. It has now been scored on all
20,672 (district × crop × season × year) cells the panel covers.

**Finding 1 — the gate genuinely discriminates.** This is the good news and it
should be reported as a result in its own right:

| Gate class | cells | share actually planted | mean area share |
|:--|--:|--:|--:|
| S1 highly suitable | 1,656 | **73.5%** | 0.070 |
| S2 suitable | 1,808 | 65.7% | 0.103 |
| S3 marginal | 1,776 | 67.2% | 0.146 |
| **N unsuitable** | 15,432 | **11.8%** | 0.026 |

A crop the gate calls suitable is **6× more likely** to actually be planted than
one it vetoes. A knowledge-based system built with no labels reproducing
farmer behaviour that strongly is a genuine, defensible result.

**Finding 2 — but the gate vetoes 74.7% of everything.** That is not a safety
net. A veto is supposed to be a rare, near-certain statement that land cannot
support a crop; at 75% the gate is doing most of the ranking, and doing it
badly at the margin.

**Finding 3 — 8.2% of vetoes are demonstrably wrong.** Of vetoed cells, 1,272
were planted with more than 1% of district area and 979 with more than 5%.
Worst offenders:

| Crop | limiting factor | false vetoes | mean area share when vetoed |
|:--|:--|--:|--:|
| **Sugarcane** | **LGP** | 46 | **1.00** ← vetoes 100% of Whole-Year area |
| Groundnut | rain | 217 | 0.47 |
| Rice | rain | 81 | 0.74 |
| Cotton(lint) | LGP | 74 | 0.41 |
| Jowar | rain | 123 | 0.37 |
| Maize | rain | 254 | 0.26 |
| Soyabean | **season** | 38 | 0.22 |

Two factors cause nearly all of it: **`rain`** and **`LGP`**.

**Finding 4 — the fine-grained ordering is inverted.** Mean area share rises
S1 (0.070) → S2 (0.103) → S3 (0.146). Among non-vetoed crops the gate's grading
runs *backwards* against practice. The veto/no-veto split works; the 4-level
grade does not.

**Finding 5 — the rainfed assumption causes a third of the damage.** Allowing
irrigation drops vetoes 74.7% → 63.1% and false vetoes 8.2% → **2.8%**.
Irrigation is currently a boolean the caller guesses at.

## The strategy

### S2.1 — Make the gate falsifiable *(do first; it gates everything else)*
Ship the diagnostic above as a permanent, versioned evaluation.

- `src/eval/gate_validation.py` → `reports/gate_validation.md`
- Metrics: veto rate, **false-veto rate at >1% and >5% area**, planted-rate by
  class, and area-share monotonicity across classes
- CI test: false-veto rate at >5% area must not regress

**Why first:** every subsequent change to an envelope is currently unmeasurable.
This turns S2 from an assertion into an instrument. It is also the most
publishable thing in the project — *validating a knowledge-based land-suitability
model against eight years of revealed cropping practice* is a real contribution.

**Target:** veto rate 74.7% → **< 35%**; false-veto (>5% area) 6.3% → **< 1.5%**.

### S2.2 — Repair the two broken factors
Not a rewrite — two specific, diagnosable bugs.

- **`LGP`**: length of growing period is a *rainfed* concept computed as days
  where P > 0.5·ET₀. Long-duration crops (sugarcane 330 d, cotton 180 d,
  turmeric 250 d, banana 330 d) can never satisfy it in a monsoon climate, so
  they are vetoed everywhere. Sugarcane is vetoed on 100% of the area it
  actually occupies. Fix: LGP must not veto perennial and long-duration crops;
  for those the constraint is *water availability over the year*, which
  `effective_water` already models.
- **`rain`**: Kharif crops are being vetoed in districts that grow them
  heavily. The absolute bounds are still too tight at the dry end for
  Groundnut, Maize, Jowar and Rice. Recalibrate `abs_min` against the observed
  distribution of rainfall where each crop is grown — **as a flag for review,
  never as an automatic replacement**, or the gate silently becomes a second
  copy of the ranker and stops being an independent check.

### S2.3 — Replace the 4-level grade with a hard/soft split
Finding 4 says the grade is not trustworthy. Distinguish two kinds of failure:

- **Hard veto** — physically impossible: insufficient soil depth, drainage
  incompatible with the crop's requirement, wrong season. These are the vetoes
  worth keeping, and they are the ones the plan §3 justifies the gate with
  ("rice on a shallow, excessively-drained upland").
- **Soft penalty** — marginal climate. Feeds the ranker as a continuous
  feature (it already does, via early fusion) but never removes a crop.

This directly addresses the over-veto without weakening the safety property.

### S2.4 — Derive irrigation from the panel *(no new data needed)*
Irrigation is the biggest missing confounder and there is no irrigation
dataset. But the APY panel contains a strong proxy: **the share of a district's
cropped area that is Rabi or Summer**. Rabi and Summer cropping in Maharashtra
is overwhelmingly irrigated; a district with 40% Rabi area has assured water,
one with 5% does not. Sugarcane area share is a second, independent proxy.

Build `irrigation_index` per district-year from the panel, validate that it
correlates with known canal-command districts, and use it to set the
`irrigated` flag instead of asking the caller to guess.

**Expected:** false-veto 8.2% → ~3% on its own (measured, from the
irrigated-gate comparison).

### S2.5 — Verify the envelopes against their sources
The worksheet already exists: `reports/envelope_verification_worksheet.csv`,
130 factor rows across 27 crops, `verified: 0`.

Do not bulk-add the remaining 51 crops first — that multiplies unverified
numbers. Verify by area coverage instead:

1. **Tier 1 (10 crops ≈ 90% of Maharashtra's cropped area):** Jowar, Cotton,
   Soybean, Sugarcane, Gram, Tur, Bajra, Rice, Wheat, Groundnut.
2. **Tier 2:** the remaining 17 encoded crops.
3. **Tier 3:** extend to the 51 uncovered fertiliser crops, with sources.

Record source and retrieved-on date per number; populate
`VERIFIED_AGAINST_SOURCE` one crop at a time.

**This is the single highest value-per-hour item in the project** and it needs
no code — it is the difference between "I chose these numbers" and "FAO and
ICAR chose these numbers" when a panel asks.

---

# Part 2 · S3 — the yield model

## The diagnosis

Current state: within-crop ρ = 0.42, R² = 0.19, conformal bands correctly
calibrated (0.900 / 0.800) but **1.76× the median wide** — honest and not
decision-useful. `src/models/yield_class.py` exists but is **orphaned**: never
wired into the pipeline, never tested.

**The finding that matters — the panel contains a strong feature I am not using.**
Lagged yield for the same district-crop-season, using strictly past values:

| Protocol | features | R² | within-crop ρ |
|:--|:--|--:|--:|
| GroupKFold — new district, no history | no lags | 0.190 | 0.420 |
| GroupKFold — new district, no history | **+ lagged yield** | **0.309** | **0.496** |
| Forward chaining — known district, next year | no lags | 0.235 | 0.465 |
| Forward chaining — known district, next year | **+ lagged yield** | **0.283** | **0.506** |

**+0.119 R²** — larger than every feature-engineering gain in the original plan
combined, and it costs nothing but a `groupby().shift()`.

## The strategy

### S3.1 — Two regimes, because they are genuinely different products
The gap between the two protocols above is not noise, it is a design signal:

- **Warm start** — a district with cropping history. Lags available. Serve the
  lagged model. Deployment-realistic gain: **+0.048 R²** (forward chaining).
- **Cold start** — a taluka or crop with no history. Lags are NaN. Serve the
  no-lag model.

Do not report the GroupKFold-with-lags number (0.309) as the headline: under
that protocol a test district's lags come from its own held-out rows, so it
flatters what a genuinely new district would get. Report it as the
*upper* bound and forward chaining as the honest one. Say so explicitly.

### S3.2 — Panel features beyond lag-1
Cheap, and all strictly backward-looking:

- `lag1`, `lag2` yield z
- expanding mean and SD of past yields (district-crop-season)
- lagged area share (is the crop expanding or contracting here?)
- `n_prior_years` — how much history exists, which is also the warm/cold switch
- district-crop historical rank among crops

### S3.3 — Wire in and validate `yield_class`
It exists and is better matched to the data than a point estimate. Finish it:

- wire into `pipeline.train()` and the serving path
- report per-class precision/recall and the confusion matrix against a
  majority-class baseline
- serve **class as primary**, band as secondary

### S3.4 — Per-crop abstention
Consistent with the S5 philosophy already in the system. Serve a yield estimate
only where cross-validated per-crop ρ clears a threshold; elsewhere return
"insufficient signal for this crop" rather than a number. Currently 2 of 28
crops sit below ρ = 0.2 — those should not be emitting yield bands at all.

### S3.5 — Per-crop models for the majors
With eight years the big crops have enough rows for their own model — Maize
649, Jowar 461, Sesamum 468, Groundnut 452. A pooled model cannot learn
crop-specific responses; a per-crop model can. Test against the pooled model
per crop and keep whichever wins, per crop.

### S3.6 — Fix the z → t/ha conversion
`crop_stats` currently uses the whole-panel crop mean and SD to convert a
predicted z-score back to tonnes. For a serving year that is unknowable.
Use past-years-only statistics, and expose the conversion's own uncertainty.

---

# Sequencing

| # | Item | Effort | Expected gain | Depends on |
|:--|:--|:--|:--|:--|
| 1 | S2.1 gate validation harness | S | makes S2 measurable | — |
| 2 | S3.2 panel/lag features | S | **+0.05 R² deployment-realistic** | — |
| 3 | S2.2 fix LGP + rain factors | M | veto 75%→~35% | 1 |
| 4 | S3.1 warm/cold two-regime serving | M | correct claims | 2 |
| 5 | S2.4 irrigation index from panel | M | false-veto 8.2%→~3% | 1 |
| 6 | S3.3 wire in `yield_class` | S | usable S3 output | — |
| 7 | S2.3 hard/soft veto split | M | removes bad grading | 1, 3 |
| 8 | S3.4 per-crop abstention | S | quality over coverage | 6 |
| 9 | S2.5 verify Tier-1 envelopes | L, no code | **defensibility** | — |
| 10 | S3.5 per-crop models | M | +ρ on majors | 2 |

Items 1, 2, 6 and 9 have no dependencies and can start immediately.

# Success criteria

**S2 → 8.5**
- veto rate < 35%, false-veto (>5% area) < 1.5%
- area share increases monotonically S3 → S2 → S1
- irrigation derived from data, not asked of the caller
- Tier-1 crops (≈90% of area) verified against cited sources

**S3 → 8.5**
- within-crop ρ ≥ 0.50 under forward chaining
- `yield_class` served and beating a majority baseline by a reported margin
- zero yield numbers emitted for crops below the skill threshold
- warm-start and cold-start reported separately and honestly

# What this deliberately does not attempt

Neither stage can be fixed by modelling alone past a point.

- **Weather still never overlaps the labels.** Crop years run 2015-16..2022-23,
  weather 2023-24..2025-26. Weather stays a climatology descriptor. Acquiring
  IMD gridded rainfall for 2015-2023 would let S3 model weather *response*, and
  is worth more than every item above.
- **Irrigation index is a proxy, not a measurement.** The Minor Irrigation
  Census remains the real fix.
- **Verification is human work.** No amount of code substitutes for checking
  a number against FAO EcoCrop.
