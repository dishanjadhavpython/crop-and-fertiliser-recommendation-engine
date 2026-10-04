# Engine scorecard

What "9.5" means for this engine, measured the same way on every run. Each sub-score maps its measured value piecewise-linearly from a frozen floor (0) to a frozen target (10); the composite is their weighted mean. Any failing hard gate caps the composite at 6.0. Unmeasured sub-scores are shown, never silently scored. Anchors sha256 `252d153062676072` (frozen).

## Latest: Phase 4 re-run: test subprocesses given the engine packages — 2026-10-04 00:18

**Composite 7.96 / 10** (uncapped 7.96; 90% of the weight measured). Run time 30.2 min.

### Hard gates

| Gate                                | Status   | Detail                                                                                                |
|:------------------------------------|:---------|:------------------------------------------------------------------------------------------------------|
| all tests green                     | pass     | 257 passed, 37 warnings in 1155.07s (0:19:15)                                                         |
| fertiliser table reproduced exactly | pass     | 1000/1000 sampled keys reproduce the table exactly                                                    |
| irrigation never adds a veto        | pass     | 0 (taluka, season, crop) cells are vetoed only when irrigated                                         |
| photo never removes a veto          | pass     | 2304 fusions checked over 48 surveyed pairs and 8 classes; none moved depth, drainage, salinity or pH |
| no-leakage tests pass               | pass     | 12 passed in 158.71s (0:02:38)                                                                        |
| cache key covers every input        | pass     | 93/93 inputs hashed                                                                                   |

### Sub-scores

| Sub-score                                               | Measured   |   0 at |   10 at |   Weight | Score   |
|:--------------------------------------------------------|:-----------|-------:|--------:|---------:|:--------|
| Forward-chaining NDCG@5 margin over popularity          | 0.1120     |  0.050 |   0.160 |    1.500 | 5.64    |
| Grouped + temporal NDCG@5 margin over popularity        | 0.0630     |  0.000 |   0.090 |    1.500 | 7.00    |
| False vetoes on >5% of district area                    | 0.0084     |  0.030 |   0.005 |    1.000 | 8.64    |
| Metamorphic relations (M1-M3, M5, M6) pass rate         | 0.9997     |  0.900 |   1.000 |    1.500 | 9.97    |
| Fertiliser season/irrigation context-match (M4)         | 1.0000     |  0.800 |   1.000 |    1.000 | 10.00   |
| Cold-start within-crop yield rho                        | 0.4320     |  0.300 |   0.550 |    0.750 | 5.28    |
| Served-regime 90% interval coverage error               | 0.0006     |  0.100 |   0.020 |    0.750 | 10.00   |
| Soil photo macro-F1 on the owner's dataset (grouped CV) | —          |  0.600 |   0.850 |    0.500 | —       |
| Soil photo calibration (ECE) + non-soil rejection       | —          |  0.000 |   1.000 |    0.500 | —       |
| Inputs that measurably change the answer (of 15)        | 0.9333     |  0.000 |   1.000 |    0.500 | 9.33    |
| Explanations, provenance and Maharashtra crop count     | 0.6000     |  0.000 |   1.000 |    0.500 | 6.00    |

### Ranking — the list as served

**Forward chaining (train past -> rank next year)** — 544 queries, 34 districts

| Method                                   |   ndcg@5 |   ndcg@3 |   p@3 |   recall@5 |
|:-----------------------------------------|---------:|---------:|------:|-----------:|
| Popularity prior (same protocol)         |    0.717 |    0.705 | 0.547 |      0.729 |
| District persistence (last year's share) |    0.819 |    0.832 | 0.644 |      0.785 |
| S2 rules only                            |    0.589 |    0.557 | 0.420 |      0.611 |
| Ranker alone (learned score)             |    0.844 |    0.856 | 0.653 |      0.811 |
| Engine as served (blend + veto)          |    0.829 |    0.844 | 0.642 |      0.792 |

Engine vs popularity over districts: Δ +0.1128 (95% CI +0.0902 … +0.1351), Wilcoxon p = 0, 34 districts.

**Grouped + temporal (unseen district AND year)** — 272 queries, 34 districts

| Method                           |   ndcg@5 |   ndcg@3 |   p@3 |   recall@5 |
|:---------------------------------|---------:|---------:|------:|-----------:|
| Popularity prior (same protocol) |    0.710 |    0.708 | 0.574 |      0.701 |
| S2 rules only                    |    0.601 |    0.566 | 0.440 |      0.607 |
| Ranker alone (learned score)     |    0.790 |    0.800 | 0.630 |      0.753 |
| Engine as served (blend + veto)  |    0.773 |    0.781 | 0.608 |      0.733 |

Engine vs popularity over districts: Δ +0.0632 (95% CI +0.0419 … +0.0846), Wilcoxon p = 0, 34 districts.

District persistence under forward chaining (grow what the district grew last year): NDCG@5 0.819, against the engine's 0.829.

### Metamorphic relations

34 talukas (one per district) × Kharif and Rabi; 3981 applicable checks.

| Relation   |   Checks |   Passed | Example failure                              |
|:-----------|---------:|---------:|:---------------------------------------------|
| M1         |     1176 |     1175 | AMBEGAON Kharif Rice: learned 5.735 -> 5.740 |
| M2         |     1272 |     1272 |                                              |
| M3         |       68 |       68 |                                              |
| M4         |      280 |      280 |                                              |
| M5         |      544 |      544 |                                              |
| M6         |      641 |      641 |                                              |

### Inputs that change the answer

| input      | kind   |   talukas_changed |   of | effective   |
|:-----------|:-------|------------------:|-----:|:------------|
| N          | card   |                 6 |    6 | True        |
| P          | card   |                 6 |    6 | True        |
| K          | card   |                 6 |    6 | True        |
| OC         | card   |                 6 |    6 | True        |
| pH         | card   |                 6 |    6 | True        |
| EC         | card   |                 6 |    6 | True        |
| S          | card   |                 6 |    6 | True        |
| Zn         | card   |                 6 |    6 | True        |
| Fe         | card   |                 6 |    6 | True        |
| Mn         | card   |                 6 |    6 | True        |
| Cu         | card   |                 6 |    6 | True        |
| B          | card   |                 6 |    6 | True        |
| irrigation | farm   |                 5 |    6 | True        |
| season     | farm   |                 6 |    6 | True        |
| soil photo | photo  |                 0 |    6 | False       |

### Explanation checklist

| Item                                                   |   Value |
|:-------------------------------------------------------|--------:|
| every crop carries a reason                            |   1.000 |
| every assessed crop carries its factor scores          |   1.000 |
| every crop carries its climate/soil requirement ranges |   0.000 |
| every card reading carries its own source              |   1.000 |
| recommendable Maharashtra crops (19; 19 -> 45)         |   0.000 |

### Yield

Cold-start within-crop rho 0.432. Served 90% intervals cover 0.900 (cold, GroupKFold) and 0.901 (warm, forward chaining).

### Gate

False-veto rate (>5% of district area): 0.0084; veto rate 0.613 over 20672 cells.

### Soil photo

not measured — soil-photo training waits for the owner's Maharashtra dataset (research and plan/ML_PLAN.md §6)

## History

| timestamp        | label                                                                  |   composite |   composite_uncapped |   measured_weight | gates   | anchors    |   fc_margin |   gt_margin |   false_veto |   metamorphic |   fert_context |   cold_rho |   coverage_error |   soil_f1 |   soil_calibration |   inputs_effective |   explanation |
|:-----------------|:-----------------------------------------------------------------------|------------:|---------------------:|------------------:|:--------|:-----------|------------:|------------:|-------------:|--------------:|---------------:|-----------:|-----------------:|----------:|-------------------:|-------------------:|--------------:|
| 2026-09-12 10:26 | Phase 0 baseline                                                       |       6.000 |                6.530 |             0.900 | 4/6     | frozen now |       0.111 |       0.060 |        0.009 |         1.000 |          0.730 |      0.396 |            0.004 |       nan |                nan |              0.933 |         0.400 |
| 2026-09-12 11:08 | Phase 1 correctness fixes                                              |       7.750 |                7.750 |             0.900 | 6/6     | frozen     |       0.111 |       0.060 |        0.009 |         1.000 |          1.000 |      0.396 |            0.001 |       nan |                nan |              0.933 |         0.600 |
| 2026-09-12 11:26 | Phase 1b: duplicate columns dropped, served ranker bagged + monotone   |       7.850 |                7.850 |             0.900 | 6/6     | frozen     |       0.112 |       0.066 |        0.009 |         1.000 |          1.000 |      0.387 |            0.001 |       nan |                nan |              0.933 |         0.600 |
| 2026-09-12 12:23 | Phase 2-3: 29 weather years, as-of normals, year-matched yield weather |       7.970 |                7.970 |             0.900 | 6/6     | frozen     |       0.112 |       0.063 |        0.008 |         1.000 |          1.000 |      0.432 |            0.001 |       nan |                nan |              0.933 |         0.600 |
| 2026-10-03 23:40 | Phase 4: 8-class soil fusion + card EC caution                         |       6.000 |                7.960 |             0.900 | 4/6     | frozen     |       0.112 |       0.063 |        0.008 |         1.000 |          1.000 |      0.432 |            0.001 |       nan |                nan |              0.933 |         0.600 |
| 2026-10-04 00:18 | Phase 4 re-run: test subprocesses given the engine packages            |       7.960 |                7.960 |             0.900 | 6/6     | frozen     |       0.112 |       0.063 |        0.008 |         1.000 |          1.000 |      0.432 |            0.001 |           |                    |              0.933 |         0.600 |
