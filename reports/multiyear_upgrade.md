# The multi-year upgrade

The original plan named one thing as worth more than every modelling decision
combined:

> "Get multi-year APY from data.gov.in. It is a single download, it requires no
>  new modelling, and it multiplies your effective sample size by more than
>  twenty. Every weak result in §7 is a symptom of n=34, and this is the direct
>  cure."

That data has now arrived. This report records what it changed, what it did
*not* change, and one delivered file that had to be rejected.

## What the panel became

| | Before | After |
|:--|--:|--:|
| APY years | 1 (2022-23) | **8** (2015-16 → 2022-23) |
| District-year label units | 34 | **272** |
| Labelled crop rows | 1,000 | **7,035** |
| Ranking queries | 136 | **1,088** |
| Weather years | 1 | **3** |
| Soil Health Card cycles | 1 | **3** |
| Distinct crops | 25 | 28 |
| Feature count | 165 | **305** |

---

## Administrative geography — the trap in any Indian district panel

Districts are not constant over time, and a join that assumes they are will
compare a 2015 district with a differently-shaped 2022 one. Everything below
was **verified against the delivered files**, not assumed. The registry lives
in `src/data/admin_changes.py`.

### 1. Palghar — the one that would have broken the panel

Palghar district was created on **1 August 2014** by splitting Thane. Had the
panel begun before 2014-15, Thane's area and production would drop
discontinuously in one year and every model would read that administrative
event as an agronomic collapse.

**Verified:** Palghar appears as its own district in all eight years, alongside
Thane. The panel begins *after* the split, so no reconstruction is needed.

### 2. Three district renames (2023)

| Historic name | Current name |
|:--|:--|
| Ahmednagar | Ahilyanagar |
| Osmanabad | Dharashiv |
| Aurangabad | Chhatrapati Sambhajinagar |

**Verified:** the delivered files already use the post-2023 names in *every*
year, including 2015-16 — they are retro-harmonised. The mapping is kept anyway
so that any externally sourced file is normalised on the way in and cannot
silently create a 35th district. Applying it twice would be its own bug, which
is why the verification mattered before the code.

### 3. Mumbai Suburban

Reports agriculture in only 2 of the 8 years and has no Soil Health Card record
at all. Excluded from the modelling universe as urban, which is why the
district count reads a constant 34 rather than a jittering 34/35.

### 4. Soil Health Card coverage grew, but no taluka was created

| Cycle | Talukas |
|:--|--:|
| 2023-24 | 347 |
| 2024-25 | 350 |
| 2025-26 | 351 |

The four that appear late are **Himayatnagar** and **Mudkhed** (Nanded), and
**Ghatanji** and **Kelapur** (Yavatmal).

**Verified:** all four are present in the static soil-type file and in the
daily weather, so they existed throughout — they were simply not sampled in the
earlier cycle. Their sample counts (768–1,409) sit well below the state median
of 2,380, consistent with newer partial surveys.

This distinction is load-bearing. The correct treatment is *"no measurement
yet"*, **not** *"did not exist"*: the taluka is flagged with its onset cycle,
the trend features record how many cycles it actually has, and the existing
sample-count weighting already discounts it. Treating a coverage gap as a
creation event would have taught the model that these places appeared from
nowhere in 2024.

---

## One delivered file was rejected

The file named `maharashtra_daily_weather_taluka_2022-04-01_to_2023-03-31.csv`
is a **byte-identical duplicate** of the 2025-26 file — same MD5
(`49aa65ff…`) — and its own `Date` column reads `2025-04-01 .. 2026-03-31`.

It is excluded. Including it would have done two kinds of damage: double-count
one weather year in every climatology average, and — far worse — manufacture a
false claim that weather is contemporaneous with the 2022-23 crop labels, since
the filename implies exactly the overlap the project has always lacked.

**Consequence:** three genuine weather years remain (2023-24 → 2025-26), and
they still do **not** overlap the crop panel (2015-16 → 2022-23). Weather
therefore remains strictly a *climatology descriptor*, exactly as the original
plan required. A test now asserts every weather file's internal date span
matches its filename, so this cannot recur silently.

What three years *do* buy over one: an honest **normal** instead of a one-year
snapshot, and the first measure of **inter-annual variability** — new features
that a single year cannot produce at all. Observed rainfall CV across talukas
ranges 0.02 to 0.27.

---

## Results

All protocols group by district. Two are newly possible.

### S3 — yield

| Panel | Protocol | n labels | R² | within-crop ρ |
|:--|:--|--:|--:|--:|
| 1 year | GroupKFold(district) | 1,000 | 0.164 | 0.276 |
| **8 years** | GroupKFold(district) | 7,035 | 0.191 | **0.426** |
| 8 years | forward chaining *(train past → test next year)* | 3,807 | **0.236** | **0.467** |
| 8 years | grouped + temporal *(unseen district AND year)* | 2,001 | 0.123 | 0.367 |

Within-crop ρ — the metric the plan says to report, and the one that governs
the recommender — rose **0.276 → 0.426, a 54% improvement**.

### The two crops the plan said were broken are fixed

The plan singled out cotton and gram as worse than predicting the mean, and
attributed it to missing irrigation data. More label-years fixed both without
any new variable:

| Crop | ρ, 1 year | ρ, 8 years |
|:--|--:|--:|
| Cotton (lint) | **−0.33** | **+0.375** |
| Gram | **−0.28** | **+0.285** |

Crops where the model is worse than useless (ρ < 0.2) fell from **10 of 25 to
2 of 28**. Only safflower (−0.13) and Rapeseed & Mustard (0.16) still need the
rule scorer to carry them.

### S1 — ranking

| Panel | Protocol | Popularity prior | LambdaMART + fit | Margin |
|:--|:--|--:|--:|--:|
| 1 year | GroupKFold | 0.790 | 0.873 | +0.083 |
| 8 years | GroupKFold | 0.740 | 0.862 | **+0.122** |
| 8 years | forward chaining | 0.740 | **0.888** | **+0.148** |
| 8 years | grouped + temporal | 0.757 | 0.805 | +0.048 |

The absolute NDCG@5 under GroupKFold dips slightly (0.873 → 0.862) because the
eight-year task is harder — 28 crops instead of 25, and cropping patterns that
shift across the decade. **The margin over the baseline is what matters, and it
widened from +0.083 to +0.122.** Recall@5 improved 0.841 → 0.878.

**Forward chaining is the deployment number.** Train on all history, rank next
season: **0.888**. That protocol answers the question a farmer actually asks,
and it was impossible with one year of labels.

**Grouped + temporal is the strictest honest claim.** A fold shares neither a
district nor a year with its training set, so neither spatial autocorrelation
nor an era effect can carry it. The model still clears the baseline, by +0.048.

---

## A metric bug this exposed

The first multi-year run showed Recall@5 collapsing from 0.841 to 0.301 for the
model **and for the baseline** — a change too large and too symmetric to be a
model effect.

The cause was in the evaluation, not the engine: `ranking_report` defaulted to
grouping queries by `(District, Season)`, so all eight years were pooled into
one enormous query. With ~25 relevant crops competing for 5 slots, Recall@5 is
capped near 0.2 no matter how good the ranking is.

Query identity is now derived from the frame (`query_columns`), so `Year` joins
the key automatically on a panel — for the metric *and* for the LightGBM group
boundaries, which had the same latent flaw.

---

## What did not change

**Weather is still climatology.** The crop panel and the weather years do not
overlap, so no model here can learn a dry-year/wet-year response. That was the
original plan's §2 fact 3 and it survives intact.

**The fertiliser engine is untouched.** It was already exact, and eight years of
crop statistics say nothing about a deterministic dose table.

**Irrigation is still missing.** It remains the largest absent confounder, even
though more label-years repaired cotton and gram on their own.
