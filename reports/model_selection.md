# Model selection

LightGBM, CatBoost and TabPFN v2 on identical folds, features and seeds. Significance is a paired test over the 34 districts, Holm-corrected; a challenger replaces the incumbent only on a significant win inside the 1s serving budget, and a tie goes to the model already in place. TabPFN is pinned to its v2 weights, the only version whose licence permits commercial use.

## Ranking — forward chaining

| Model    | Status   |   NDCG@5 |    Fit s |   Latency s |   Δ vs incumbent |   Holm p | Note                                                                                                                                   |
|:---------|:---------|---------:|---------:|------------:|-----------------:|---------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| lightgbm | ok       |   0.8830 |  14.5000 |      0.0016 |         nan      | nan      |                                                                                                                                        |
| catboost | ok       |   0.8860 |  78.8000 |      0.0039 |           0.0038 |   0.0210 |                                                                                                                                        |
| tabpfn   | failed   | nan      | nan      |    nan      |         nan      | nan      | TimeoutError: the arm produced nothing within its 6 min budget and was terminated; recorded rather than allowed to hang the tournament |

## Yield — cold start, GroupKFold by district

| Model    | Status    |   Within-crop rho |   Pinball p50 |   80% coverage |    Fit s | Note                                                                                                                                   |
|:---------|:----------|------------------:|--------------:|---------------:|---------:|:---------------------------------------------------------------------------------------------------------------------------------------|
| lightgbm | ok        |            0.4180 |        0.3257 |         0.7150 |  19.5000 |                                                                                                                                        |
| catboost | ok        |            0.4690 |        0.3150 |         0.6200 |  60.3000 |                                                                                                                                        |
| tabpfn   | abandoned |          nan      |      nan      |       nan      | nan      | TimeoutError: the arm produced nothing within its 6 min budget and was terminated; recorded rather than allowed to hang the tournament |

## Chosen

- **ranker**: catboost — +0.0038 NDCG@5 over lightgbm, Holm p=0.021, CI excludes zero, latency 0.0039s
- **yield**: lightgbm — catboost led by 0.051 rho but its 80% interval covers 0.620 against 0.715; a sharper point estimate is not worth an interval that overstates its confidence, so the incumbent is kept
