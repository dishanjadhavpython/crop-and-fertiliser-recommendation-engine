# Regur — a crop & fertiliser recommendation engine for Maharashtra

*Draft LinkedIn post. Every number below is measured, not estimated — each one
is reproducible from the repository with a single command.*

---

## The post

**I built a crop and fertiliser recommendation engine for Maharashtra. The most
useful thing I learned was how easy it is to report a number that means nothing.**

A model predicting crop yield from soil and weather scores **R² = 0.93**. That
looks like a finished project.

Then I gave a second model *only the crop name and the season* — no soil, no
rainfall, no pH. It scored **0.91**.

The impressive number was almost entirely the model learning that sugarcane
yields 74 t/ha and sesamum yields 0.27. It had learned arithmetic about crop
names, not agronomy. Predicting *within-crop* relative yield — the honest
question — the real signal is **R² = 0.19**.

Most of what I built afterwards came from taking that seriously.

---

### What it does

You give it a taluka, a season, and optionally your own Soil Health Card
numbers. It returns:

- a **ranked shortlist of crops**, with the reason each one ranked where it did
- a **yield outlook** — below, typical, or above that crop's own norm
- an **exact fertiliser dose**, interpolated to your soil test
- a **nitrogen split schedule** based on how likely nitrogen is to leach away
- **micronutrient corrections** the government table doesn't cover

Any crop the land genuinely cannot support is removed before you see it, and the
system says which factor removed it.

---

### How the engine works

Five stages, each built with the technique that actually fits it rather than
forcing everything into one model:

**S1 · Ranker (learned).** Crop choice isn't a classification problem — there's
no single right crop, there are ten viable ones and a farmer needs an ordered
shortlist. So it's a **learning-to-rank** problem. LambdaMART trained on
revealed preference: what farmers in comparable districts actually planted.

**S2 · Agronomic gate (rules, zero labels).** FAO EcoCrop and ICAR requirement
envelopes, combined by **Liebig's law of the minimum** — a crop is only as good
as its worst factor. This vetoes what the ranker might otherwise suggest.

**S3 · Yield (learned).** Quantile regression on within-crop z-scores, plus a
three-class model. Never a point estimate, because the data doesn't support one.

**S4 · Fertiliser (lookup, deliberately not a model).** I tested all 33,169 clean
rows of the government table: **zero** have an ambiguous dose. The table is an
exact function. Training a model on it could only introduce error into something
already perfect.

**S5 · Calibration (hybrid).** Conformal prediction intervals and an
out-of-distribution guard that abstains instead of guessing.

---

### The numbers

**Crop ranking** — NDCG@5, out-of-fold, grouped by district:

| | NDCG@5 |
|:--|--:|
| Random | 0.153 |
| Agronomic rules alone (no labels at all) | 0.553 |
| **Popularity baseline — the bar to beat** | **0.740** |
| **Full engine** | **0.883** |

That popularity baseline matters. "Recommend whatever is most planted in the
state" is trivial and scores 0.740. Most published crop-recommendation work
never computes it. Beating it by **+0.143** is a claim that means something;
a bare accuracy figure isn't.

**Agronomic gate**, validated against 20,672 real district-crop-season-year
cells: a crop it calls suitable is **6.4× more likely to have actually been
planted** than one it vetoes. False-veto rate: **0.90%**.

**Yield:** within-crop ρ = **0.50** for a district with cropping history,
**0.43** for one without. Reported separately, because those are different
products.

**Calibration:** the 90% prediction interval covers **90.0%** of held-out
districts, and the 80% covers **80.0%**. Exactly on nominal.

**Fertiliser:** 1,000 sampled keys reproduce the government table byte-for-byte.
Verified in CI on every commit.

---

### The data

Five public sources, joined at taluka grain:

- **Soil Health Card** — 351 talukas × 12 soil components, 3 survey cycles
- **Soil type** — texture, depth, drainage, taxonomy per taluka
- **Daily weather** — 3 agricultural years, ~131,000 rows per year
- **Crop statistics** — 8 years (2015-16 → 2022-23), **7,035 district-crop-season
  records**
- **Fertiliser recommendations** — 33,169 clean rows across 78 crops

That becomes a feature store of **351 talukas × 305 engineered features**, and a
training panel of **272 district-years**.

Getting the panel right mattered more than any model choice. Districts are not
constant across eight years: **Palghar was carved out of Thane in 2014**, and
Ahmednagar, Osmanabad and Aurangabad were all renamed in 2023. A join that
ignores that silently compares two different shapes of district and reads an
administrative event as an agronomic one.

I also found a delivered weather file whose name said 2022-23 but whose contents
were a byte-identical copy of 2025-26. Using it would have manufactured a false
claim that the weather was contemporaneous with the crop labels. It's excluded,
and a test now asserts every file's internal dates match its name.

---

### Techniques I hadn't used before

- **Learning-to-rank (LambdaMART / lambdarank)** — the single highest-impact
  decision. Reframing recommendation as ranking rather than regression was worth
  more than every hyperparameter I touched.
- **Conformal prediction** — distribution-free intervals with a coverage
  guarantee. Turns a weak model into an honest one.
- **Compositional data analysis (isometric log-ratio)** — Soil Health Card values
  are percentages that sum to 100, so feeding all three parts to a model creates
  collinearity by construction. ILR is the correct treatment.
- **Hargreaves ET₀ (FAO-56)** — no solar radiation in the data, so
  Penman-Monteith was out. Validated against FAO's published worked example:
  32.19 vs 32.2 MJ/m²/day.
- **Mahalanobis distance for out-of-distribution detection** — so the system can
  say "this place is unlike anything I trained on" instead of guessing.
- **Linear programming** (`scipy.optimize.linprog`) for least-cost fertiliser
  blending.
- **Forward-chaining / temporal cross-validation** — train on the past, test on
  the next year, which is the question a deployed recommender actually faces.

**Stack:** Python 3.13 · LightGBM · scikit-learn · scipy · pandas · FastAPI ·
vanilla JS with hand-built SVG data visualisation. **178 tests.**

---

### The result I'm least proud of, and most glad I checked

I built a linear program to find the cheapest fertiliser mix meeting a nutrient
target. Across 400 sampled cases it **never beat the government's own published
option by more than ₹0.23 per hectare**.

With only four straight fertilisers, the existing recommendation is already
cost-optimal. The optimiser only earns its place when the table has no answer at
all — a product is locally unavailable, or sulphur is needed.

I reported it as a negative result rather than quietly dropping it. A project
where everything worked is usually a project where not enough was measured.

---

### What it still can't do

The weather years and the crop years don't overlap, so weather is a
*climatology descriptor* — "what this place is typically like" — never a
weather-response model. There's no irrigation variable, which is the single
biggest missing confounder. And the agronomic envelopes are encoded from
FAO and ICAR references but **not yet verified line by line against the primary
sources** — which is the next thing I'm doing, and no amount of code substitutes
for it.

---

*#MachineLearning #Agriculture #DataScience #Python #AgriTech #LearningToRank
#Maharashtra*

---

## Notes before you post

- **Trim for LinkedIn.** The platform truncates around 1,300 characters. The
  strongest opening is the R² = 0.93 vs 0.91 story — that alone is a full post
  if you want a shorter one.
- **Lead with a screenshot of the console.** The map with 351 dots shaded by
  aridity is the most arresting single image the project has.
- **Every figure here is reproducible.** If someone challenges a number in the
  comments: `pytest tests -q` and `python -m src.eval.ablation` regenerate all
  of them.
- **Don't claim the envelopes are verified.** They aren't yet, and the post says
  so — keep it that way.
