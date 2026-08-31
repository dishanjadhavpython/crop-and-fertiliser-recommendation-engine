# Benchmark results

Reproduces the tables in §7 of the project plan, on this repository's own feature store and evaluation harness.

Protocol: GroupKFold by district (6 folds), 3-seed bagging, 34 districts x 8 years = 272 district-year label units, 7035 rows with a recorded yield, 30464 rows in the ranking candidate grid, 1088 ranking queries.

## The headline number is a mirage

| Target                                        |    R2 | Verdict                                                      |
|:----------------------------------------------|------:|:-------------------------------------------------------------|
| Raw yield (t/ha), soil + weather features     | 0.822 | the mirage                                                   |
| Raw yield (t/ha), crop + season identity ONLY | 0.787 | almost the same — the model learned crop scale, not agronomy |
| Within-crop z-scored yield, soil + weather    | 0.191 | the honest signal                                            |

## Why GroupKFold by district is not optional

| Protocol                                            |    R2 |
|:----------------------------------------------------|------:|
| Random KFold (LEAKS — districts split across folds) | 0.325 |
| GroupKFold by district (6 folds)                    | 0.191 |
| Leave-one-district-out (34 folds)                   | 0.163 |
| Spatial block CV (KMeans on centroids)              | 0.053 |

With eight years of labels two further protocols become possible, and they are the ones that matter for deployment. **Forward chaining** trains on the past and ranks the next season — the question a farmer actually asks. **Grouped + temporal** shares neither a district nor a year between train and test, so neither spatial autocorrelation nor an era effect can carry a result. Both are reported in `reports/multiyear_upgrade.md`.

## 7.1 Feature block ablation

| Feature set                        |     R2 |   Delta |   Per-crop rho |   n feat |
|:-----------------------------------|-------:|--------:|---------------:|---------:|
| Null model (predict the mean)      |  0.000 | nan     |        nan     |        0 |
| Crop + season identity only        | -0.012 |  -0.012 |         -0.160 |        2 |
| + Soil Health Card (12 components) |  0.107 |   0.119 |          0.285 |       49 |
| + Soil physical                    |  0.161 |   0.054 |          0.376 |       66 |
| + Agro-climatic (engineered)       |  0.182 |   0.021 |          0.400 |      147 |
| + Agronomic interactions           |  0.191 |   0.009 |          0.426 |      310 |

### Blocks the n=34 constraint rejects

| Feature set             |    R2 |   n feat |
|:------------------------|------:|---------:|
| Best set (blocks A-D)   | 0.191 |      310 |
| + Spatial kNN smoothing | 0.174 |      326 |
| + Contextual z-scores   | 0.188 |      324 |

## 7.2 Model-selection levers

| Lever                                    |    R2 |
|:-----------------------------------------|------:|
| Baseline, 310 features                   | 0.191 |
| Nested importance selection -> top 35    | 0.170 |
| Stronger regularisation (leaves=7, L2=5) | 0.146 |
| Very strong regularisation (leaves=4)    | 0.111 |
| DART boosting                            | 0.129 |
| Seed bagging, 1 seed(s)                  | 0.191 |
| Seed bagging, 10 seed(s)                 | 0.191 |

## 7.3 Ranking benchmark  <- the headline result

| Method                                           |   ndcg@5 |   ndcg@3 |   p@3 |   recall@5 |
|:-------------------------------------------------|---------:|---------:|------:|-----------:|
| Random ranking                                   |    0.153 |    0.125 | 0.112 |      0.186 |
| Yield-regression framing                         |    0.730 |    0.680 | 0.459 |      0.761 |
| Area-share regression framing                    |    0.798 |    0.800 | 0.552 |      0.783 |
| S2 agronomic rules only (no labels)              |    0.553 |    0.526 | 0.377 |      0.592 |
| LambdaMART, crop identity only                   |    0.412 |    0.340 | 0.245 |      0.548 |
| Popularity prior, out-of-fold  <- the bar        |    0.740 |    0.712 | 0.528 |      0.804 |
| LambdaMART + engineered features                 |    0.864 |    0.855 | 0.610 |      0.865 |
| Full engine (early fusion + blend)  <- as served |    0.883 |    0.873 | 0.622 |      0.879 |

## 7.4 Where the model works and where it fails

| Crop                |   n |    rho |   mae_t_ha |
|:--------------------|----:|-------:|-----------:|
| Tobacco             |   9 |  0.730 |      0.653 |
| Rice                | 299 |  0.727 |      0.559 |
| Groundnut           | 452 |  0.668 |      0.354 |
| Niger seed          | 138 |  0.644 |      0.059 |
| Maize               | 649 |  0.627 |      0.649 |
| Sugarcane           | 220 |  0.577 |     13.511 |
| Small millets       |  58 |  0.548 |      0.239 |
| Wheat               | 244 |  0.543 |      0.333 |
| Sesamum             | 468 |  0.513 |      2.806 |
| Soyabean            | 281 |  0.506 |      0.358 |
| Ragi                | 110 |  0.498 |      0.289 |
| Other Kharif pulses | 229 |  0.486 |      0.140 |
| Other Summer Pulses | 117 |  0.485 |      0.142 |
| Bajra               | 196 |  0.475 |      0.255 |
| Other Cereals       | 441 |  0.469 |      0.243 |
| Jowar               | 461 |  0.435 |      0.277 |
| Arhar/Tur           | 262 |  0.415 |      0.277 |
| Sunflower           | 348 |  0.375 |      0.199 |
| Cotton(lint)        | 212 |  0.375 |      0.815 |
| Gram                | 259 |  0.285 |      0.201 |
| Castor seed         |  80 |  0.285 |      0.103 |
| Urad                | 287 |  0.279 |      0.148 |
| Moong(Green Gram)   | 312 |  0.260 |      0.155 |
| Linseed             | 107 |  0.257 |      0.120 |
| other oilseeds      | 317 |  0.215 |      0.200 |
| Other Rabi pulses   | 227 |  0.211 |      0.165 |
| Rapeseed &Mustard   | 104 |  0.163 |      0.167 |
| Safflower           | 148 | -0.130 |      0.181 |

### Resulting per-crop blend weights (S5)

| Crop                |   n |    rho |   alpha | decided_by   |
|:--------------------|----:|-------:|--------:|:-------------|
| Tobacco             |   9 |  0.730 |   1.000 | model        |
| Rice                | 299 |  0.727 |   1.000 | model        |
| Groundnut           | 452 |  0.668 |   1.000 | model        |
| Niger seed          | 138 |  0.644 |   1.000 | model        |
| Maize               | 649 |  0.627 |   1.000 | model        |
| Sugarcane           | 220 |  0.577 |   0.940 | model        |
| Small millets       |  58 |  0.548 |   0.870 | model        |
| Wheat               | 244 |  0.543 |   0.860 | model        |
| Sesamum             | 468 |  0.513 |   0.780 | blend        |
| Soyabean            | 281 |  0.506 |   0.770 | blend        |
| Ragi                | 110 |  0.498 |   0.740 | blend        |
| Other Kharif pulses | 229 |  0.486 |   0.710 | blend        |
| Other Summer Pulses | 117 |  0.485 |   0.710 | blend        |
| Bajra               | 196 |  0.475 |   0.690 | blend        |
| Other Cereals       | 441 |  0.469 |   0.670 | blend        |
| Jowar               | 461 |  0.435 |   0.590 | blend        |
| Arhar/Tur           | 262 |  0.415 |   0.540 | blend        |
| Sunflower           | 348 |  0.375 |   0.440 | blend        |
| Cotton(lint)        | 212 |  0.375 |   0.440 | blend        |
| Gram                | 259 |  0.285 |   0.210 | blend        |
| Castor seed         |  80 |  0.285 |   0.210 | blend        |
| Urad                | 287 |  0.279 |   0.200 | blend        |
| Moong(Green Gram)   | 312 |  0.260 |   0.150 | blend        |
| Linseed             | 107 |  0.257 |   0.140 | blend        |
| other oilseeds      | 317 |  0.215 |   0.040 | blend        |
| Other Rabi pulses   | 227 |  0.211 |   0.030 | blend        |
| Rapeseed &Mustard   | 104 |  0.163 |   0.000 | rules        |
| Safflower           | 148 | -0.130 |   0.000 | rules        |

## 5.4 Conformal calibration

| Interval                        |   coverage |   mean_width |    n |
|:--------------------------------|-----------:|-------------:|-----:|
| Raw quantile model, nominal 80% |      0.728 |        1.851 | 7035 |
| Split-conformal, nominal 90%    |      0.900 |        2.742 | 7035 |
| Split-conformal, nominal 80%    |      0.800 |        2.118 | 7035 |

## 6.5 Does the least-cost LP save money?

An honest negative result. With only four straight fertilisers (Urea, DAP, SSP, MOP) and three nutrient constraints, the government table's Option 1 is already cost-optimal at standard subsidised prices — the LP recovers it to within a rupee and never beats it. The optimiser earns its place only where the table cannot help: when a product is locally unavailable, or when the micronutrient layer requires sulphur that no published option supplies.

| Scenario                                      |   Mean cost delta vs table (INR/ha) |   Max saving found (INR/ha) |   LP solved |   n |
|:----------------------------------------------|------------------------------------:|----------------------------:|------------:|----:|
| LP vs cheapest published option               |                              -0.040 |                       0.230 |         400 | 400 |
| DAP locally unavailable — table has no answer |                             268.560 |                     nan     |         400 | 400 |
| 20 kg S/ha required — table has no answer     |                             267.430 |                     nan     |         368 | 400 |
