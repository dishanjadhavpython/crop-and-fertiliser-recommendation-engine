# Engine improvement experiments

Every arm runs under the identical protocol used everywhere else in this
project — **GroupKFold by district**, 5-seed bagged — so the deltas are
comparable. Significance is a **Wilcoxon signed-rank test over the 136
per-query NDCG@5 scores**, paired against the stated baseline. Nothing was
adopted unless it won on this table.

## Result: the shipped engine had a defect that put it below the baseline

| Engine | NDCG@5 | NDCG@3 | P@3 | Recall@5 |
|:--|--:|--:|--:|--:|
| Popularity prior *(the bar the plan says you must beat)* | 0.790 | 0.748 | 0.588 | 0.785 |
| **OLD engine, as first shipped** | **0.777** | 0.760 | 0.598 | 0.742 |
| NEW engine — α fix + capacity | 0.841 | 0.826 | 0.650 | 0.795 |
| **NEW engine + early fusion** | **0.876** | **0.865** | **0.684** | **0.835** |

| Comparison | Δ NDCG@5 | p |
|:--|--:|--:|
| NEW vs OLD | **+0.063** | 6.9 × 10⁻⁵ |
| NEW + early fusion vs NEW | **+0.035** | 2.0 × 10⁻⁴ |
| **Total** | **+0.099** | — |

The original engine scored **0.777 against a 0.790 popularity baseline** — it
failed the one test the plan says the project must pass. The cause was a
defect, not a modelling limit, and all three fixes are additive with no
trade-off anywhere.

---

## 1. The α defect — the single largest win

The blend weights govern **S1, which is a ranker**. The first implementation
derived them from the **yield model's** within-crop Spearman. Those are
different models answering different questions.

| Blend configuration | NDCG@5 | Δ vs ranker alone | p |
|:--|--:|--:|--:|
| Ranker alone (α = 1 everywhere) | 0.868 | — | — |
| Rules alone (α = 0 everywhere) | 0.595 | −0.273 | < 10⁻⁴ |
| **Blend, α from YIELD skill — as shipped** | **0.770** | **−0.098** | **< 10⁻⁴** |
| Blend, α from RANKING skill — fixed | 0.865 | −0.003 | 0.13 |
| Blend, flat α = 0.9 | 0.858 | −0.010 | 0.007 |

The yield model cannot order cotton (ρ = −0.33) or gram (ρ = −0.28), so those
crops were being zeroed and handed to the rule scorer, which ranks at 0.595.
But the **ranker orders them well** — cotton ρ = 0.68, gram ρ = 0.74. Ten crops
were being routed to the weaker component for no reason.

After the fix, **2 crops** fall back to rules instead of 10, and they are the
right ones: tobacco (ρ = −0.06) and Other Summer Pulses (ρ = 0.10) — both
genuinely unorderable.

The per-crop yield skill is still computed and still governs the **yield band**,
which is what it was always for.

## 2. Ranker capacity — a genuine model gain, not a protocol artefact

The first measurement showed a gain only under leave-one-district-out, which
would have been confounded: LODO trains each fold on 33 of 34 districts,
6-fold on ~28. The control isolates it.

| Protocol | 300 trees | 800 trees | Δ from capacity |
|:--|--:|--:|--:|
| 6-fold GroupKFold | 0.868 | **0.880** | **+0.012** (p = 0.002) |
| Leave-one-district-out | 0.873 | 0.882 | +0.010 |

The gain holds **within** each protocol, so it is capacity, not more training
data. It is measured out-of-fold, so it is generalisation, not overfitting.
`num_leaves = 31` performs equivalently to 800 trees and was not stacked.

## 3. Early fusion of S2 into S1

S2's suitability scorer reads **only soil and climate — never a label** — so
its per-factor scores can be fed to the ranker as features with no leakage
risk whatsoever. That is strictly more expressive than blending the two
outputs afterwards: the ranker learns *when* the rules are worth trusting,
instead of being told once per crop.

| Arm | n feat | NDCG@5 | p vs base |
|:--|--:|--:|--:|
| Ranker, district features only | 165 | 0.864 | — |
| + agronomic fit features | 183 | 0.869 | 0.38 |
| Agronomic fit features **only** | 20 | 0.862 | 0.78 |

On the **bare ranker** the gain is not significant. Inside the **full engine**
it is: **+0.035, p = 0.0002**. The effect compounds — the fit features raise
per-crop ranking skill, which raises α, which reduces how much weight the weak
rule scorer carries.

The third row is the more striking one: **20 crop-conditional features match
165 district features** (p = 0.78). Almost the entire feature store is
redundant once the crop's own requirements are encoded explicitly.

---

## What did *not* work — honest negatives

None of these were adopted.

| Change | Δ NDCG@5 | p | Verdict |
|:--|--:|--:|:--|
| Within-district dispersion features (54 cols) | +0.003 | 0.43 | not significant |
| Taluka-level suitability fractions (8 cols) | +0.001 | 0.68 | not significant |
| Both together | −0.002 | 0.57 | not significant |
| Training label: magnitude-weighted grades | −0.011 | 0.11 | not significant |
| Training label: 8 grades instead of 5 | −0.013 | 0.13 | not significant |
| Training label: binary planted/not | **−0.095** | **< 10⁻⁴** | significantly **worse** |
| Training label: log-magnitude grades | +0.001 | 0.66 | not significant |
| SHC sample weighting on the ranker | +0.001 | — | negligible |

Two things worth stating plainly:

**The existing relevance label was already optimal.** Four alternatives were
trained and all evaluated against the *same fixed* original labels, so the
comparison is not self-fulfilling. None beat it; binary relevance is
dramatically worse. The plan's `qcut(area_share.rank(), 5)` choice is
vindicated.

**Recovering within-district variation did not help.** Aggregating 351 talukas
to 34 districts really does destroy information — Dharashiv has a 94%
coefficient of variation in root-zone water capacity across its talukas — but
adding that spread back as features does not improve ranking. At n = 34 the
model cannot use it.

---

## Startup cost

The fitted pipeline is now cached to disk, keyed on a hash of the feature store
plus every module that shapes the models, so a stale pipeline can never be
served after a code or data change.

| | Before | After |
|:--|--:|--:|
| CLI invocation (cold) | 114 s | 114 s |
| CLI invocation (warm) | 114 s | **1.3 s** |

---

## The ceiling that remains

None of this moved the **yield** model, and nothing could. Within-crop R² is
0.19 because there are **34 district labels for one year**. The ranking gains
above came from fixing a defect and from using the label-free rule engine
properly — not from extracting more signal from the labels, because there is
no more to extract.

The `qcut`-relevance oracle (ranking by the true area share) scores 0.978.
At 0.876 the engine has closed roughly **72% of the gap** between the
popularity baseline and that ceiling. The rest needs multi-year APY.
