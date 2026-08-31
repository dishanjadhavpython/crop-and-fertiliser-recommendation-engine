# S2 gate validation against revealed practice

The agronomic gate uses **no labels**. This report checks it against eight years of what Maharashtra's farmers actually planted (20,672 district x crop x season x year cells).

## Does the gate agree with practice?

| suitability_class   |   cells |   planted_rate |   mean_area_share |   mean_area_share_if_planted |
|:--------------------|--------:|---------------:|------------------:|-----------------------------:|
| S1                  |    2192 |          0.786 |             0.140 |                        0.178 |
| S2                  |    1848 |          0.667 |             0.125 |                        0.188 |
| S3                  |    1392 |          0.598 |             0.075 |                        0.125 |
| N                   |   15240 |          0.107 |             0.021 |                        0.192 |

A crop the gate calls suitable is **6.39x** more likely to be planted than one it vetoes. That is the gate working.

- planted rate falls monotonically S1->N: **True**
- mean area share falls monotonically S1->N: **True**

## Veto quality

The veto is decided on the *most favourable water scenario*, so it means "this land cannot support this crop" rather than "cannot without irrigation". Crops that fail only on water are reported as needing irrigation instead of being removed.

| gate                       |   scored_cells |   vetoed_cells |   veto_rate |   false_veto_planted |   false_veto_rate_planted |   false_veto_area_gt_0.01 |   false_veto_rate_area_gt_0.01 |   false_veto_area_gt_0.05 |   false_veto_rate_area_gt_0.05 |
|:---------------------------|---------------:|---------------:|------------:|---------------------:|--------------------------:|--------------------------:|-------------------------------:|--------------------------:|-------------------------------:|
| hard-veto (water-relieved) |          20672 |          12848 |       0.622 |                  408 |                     0.032 |                       187 |                          0.015 |                       115 |                          0.009 |

## The new middle category: viable, but only with irrigation

| requires_irrigation   |   cells |   planted_rate |   mean_area_share |
|:----------------------|--------:|---------------:|------------------:|
| False                 |   18280 |          0.230 |             0.037 |
| True                  |    2392 |          0.511 |             0.115 |

## Which factor is responsible (rainfed)

| limiting_factor   |   vetoes |   false_vetoes |   false_veto_rate |   mean_area_when_wrong |
|:------------------|---------:|---------------:|------------------:|-----------------------:|
| season            |    12240 |             47 |             0.004 |                  0.206 |
| rain              |      472 |             44 |             0.093 |                  0.356 |
| salinity          |       64 |             12 |             0.188 |                  0.308 |
| drainage          |       64 |              8 |             0.125 |                  0.457 |
| depth             |        8 |              4 |             0.500 |                  1.000 |

## Worst false vetoes — crops the gate rejects but farmers plant heavily

| Crop              | limiting_factor   |   n |   mean_area_share |   max_area_share |
|:------------------|:------------------|----:|------------------:|-----------------:|
| Soyabean          | season            |  38 |             0.215 |            0.815 |
| Groundnut         | rain              |  24 |             0.536 |            1.000 |
| Moong(Green Gram) | rain              |  12 |             0.146 |            0.530 |
| Jowar             | season            |   9 |             0.167 |            0.312 |
| Rice              | drainage          |   8 |             0.457 |            0.476 |
| Rice              | rain              |   8 |             0.132 |            0.199 |
| Soyabean          | salinity          |   8 |             0.408 |            0.496 |
| Sesamum           | salinity          |   4 |             0.109 |            0.155 |
| Sugarcane         | depth             |   4 |             1.000 |            1.000 |
