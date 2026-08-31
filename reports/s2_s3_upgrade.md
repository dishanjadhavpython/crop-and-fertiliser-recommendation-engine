# S2 and S3 upgrade — results

Execution of `PLAN_S2_S3.md`. Every number below is measured against the
eight-year panel under the protocol named beside it.

---

# S2 · the agronomic gate

## What the validation found

The gate uses **no labels**, so nothing had ever checked it. Scoring it on all
20,672 (district × crop × season × year) cells produced three findings, and the
first one reframed the other two.

**The season veto is 79% of all vetoes and is only 0.4% wrong.** The headline
"the gate vetoes 74.7% of everything" was my own misreading: most of that is
the gate correctly saying wheat is not a Kharif crop. The real agronomic veto
rate was 15.4%, and that is the number that mattered.

**LGP was structurally broken.** Observed length of growing period across
Maharashtra spans **70–149 days**. Sugarcane's envelope required 150. No taluka
in the state could ever satisfy it, so sugarcane was vetoed everywhere — across
**96% of the area it actually occupies**. 79% of all LGP vetoes were false.

**The rain envelopes were right; the water assumption was wrong.** Split by
season, the rain factor's false-veto rate was **2.3% in Kharif** but **38.7% in
Rabi** and **27.6% in Summer** — the two irrigated seasons. Loosening the
envelopes would have destroyed a working safety check to paper over an
irrigation problem.

## What changed

1. **LGP now gates only crops that can finish inside a rainfed season**
   (duration ≤ 150 days). Longer crops live on irrigation or stored profile
   moisture, which `effective_water` already models.
2. **Hard/soft veto split.** A veto now means *this land cannot support this
   crop*, decided on the most favourable water scenario. A crop that fails only
   on water is reported as **`requires_irrigation`** — a statement a farmer can
   act on — rather than removed.
3. **Provenance registry.** Every envelope names its species and authority
   (FAO EcoCrop plus the relevant ICAR institute), with an honest
   `VERIFIED_AGAINST_SOURCE = 0` and a printable 130-row worksheet.

## Results

| Metric | Before | After | Target |
|:--|--:|--:|--:|
| Agronomic veto rate (excl. season) | 15.4% | **2.9%** | — |
| False veto, >1% of district area | 8.24% | **1.46%** | < 2% ✅ |
| False veto, >5% of district area | 6.34% | **0.90%** | < 1.5% ✅ |
| Area share monotone S1→N | **inverted** | **correct** | monotone ✅ |
| Suitable-vs-vetoed planted ratio | 6.0× | **6.4×** | — |
| LGP false-veto rate | 79.2% | **factor no longer vetoes** | — |

Mean area share now falls correctly across classes — 0.140 → 0.125 → 0.075 →
0.021 — where before it rose. The four-level grade had been running backwards
against practice and now does not.

**The new category validates itself.** Cells flagged "viable only with
irrigation" have a **51.1% planted rate against 23.0%** elsewhere, and 3× the
mean area share. These are places where the land is fine and farmers evidently
do supply the water.

---

# S3 · the yield model

## What the diagnosis found

The panel contained a strong feature that was not being used: **lagged yield**
for the same district-crop-season, strictly backward-looking.

Two protocols, because they measure different things:

| Protocol | no lags | + panel history |
|:--|--:|--:|
| GroupKFold — new district, no history | R² 0.190 | R² 0.309 |
| Forward chaining — known district, next year | R² 0.235 | **R² 0.283** |

The 0.309 is **not** the headline. Under GroupKFold a held-out district's lags
come from its own held-out rows, so the model is handed a history a genuinely
new district does not have. It is reported as an upper bound and labelled that
way in code.

## What changed

1. **Panel history features** — lag-1, lag-2, expanding mean/SD/max of past
   yields, lagged area share and its trend, prior-year count. All built with
   `shift(1)` inside a district-crop-season group; a test asserts no row can
   see its own or a later outcome.
2. **Two regimes, served separately.** Warm start (history available) and cold
   start (none) are different products and the answer says which replied.
3. **Tercile class as the primary output.** `yield_class.py` existed but was
   orphaned — never wired in, never tested. It is now both.
4. **Per-regime abstention.** No yield number is emitted where cross-validated
   skill for that crop *in that regime* is below the floor.

## Results

| Regime | protocol | n | R² | within-crop ρ |
|:--|:--|--:|--:|--:|
| Cold start (new district) | GroupKFold by district | 7,035 | 0.191 | 0.426 |
| **Warm start (known district)** | **forward chaining** | 3,807 | **0.283** | **0.501** |
| *Upper bound — not the headline* | *GroupKFold with lags* | *7,035* | *0.306* | *0.492* |

Warm-start ρ = **0.501** clears the plan's ≥ 0.50 target.

**Tercile classifier** — the primary output:

| Regime | accuracy | majority baseline | severe-error rate |
|:--|--:|--:|--:|
| Cold start | **0.511** | 0.342 | 0.118 |
| Warm start | **0.525** | 0.341 | 0.093 |

"Severe error" is calling a below-norm district above-norm, or the reverse —
the only class confusion that would actually mislead.

**Abstention is per regime, and that mattered.** The first implementation took
`max(rho_cold, rho_warm)`, which served a crop in *both* regimes whenever
*either* was skilled. Safflower exposed it: ρ_warm = 0.263 but ρ_cold =
**−0.130**, worse than predicting the mean. A cold-start query for safflower
now returns no number.

| Regime | crops served a number | crops abstained |
|:--|--:|--:|
| Cold start | 20 of 28 | 8 |
| Warm start | 23 of 28 | 5 |

---

# Status against the plan's success criteria

**S2**
- ✅ false veto > 5% area < 1.5% → **0.90%**
- ✅ area share monotone S3 → S2 → S1
- ✅ irrigation handled without asking the caller to guess, and without feeding
  labels into a label-free gate
- ❌ Tier-1 envelope verification — worksheet ready, **0 of 27 verified**

**S3**
- ✅ within-crop ρ ≥ 0.50 under forward chaining → **0.501**
- ✅ class model served, beating majority baseline by 17–18 points
- ✅ zero yield numbers emitted below the skill threshold
- ✅ warm and cold reported separately

170 tests pass.

# What is still open

- **Envelope verification is human work** and remains undone. It is the last
  thing holding S2 back and no code substitutes for it.
- **Weather still never overlaps the labels** (crops 2015-16..2022-23, weather
  2023-24..2025-26), so S3 models climatology, not weather response.
- **Irrigation is inferred as a scenario, not measured.** The gate now asks
  "could this work with water?" rather than guessing whether water exists.
