# Phase C/D/E error analysis

The proposed ensemble does not beat all required baselines on the held-out mean normalized MAE. The improvement hypothesis is not established.

| baseline | mean_dynamic_minus_baseline | dynamic_better | tied | dynamic_worse |
| --- | --- | --- | --- | --- |
| cits | 0.022457 | 4 | 0 | 17 |
| synthetic_control | -0.045581 | 21 | 0 | 0 |
| equal_weight | 0.000303 | 0 | 7 | 14 |
| best_historical | -0.045581 | 21 | 0 | 0 |

Differences are paired by experiment; positive differences mean the proposed ensemble is worse. Win counts are descriptive, not independent Bernoulli trials.

CITS weights range from 0.495148 to 0.500000 (mean 0.497776). The selected temperature 3.0 produces nearly equal weights, explaining the small difference from equal averaging. SCM was the lowest-error estimator in training; this ranking reverses in the test averages. These observations show limited transfer of the training ranking to the test period, without identifying a causal explanation.

| split | model_name | mean_actual_error | mean_predicted_error | mean_outcome_scale |
| --- | --- | --- | --- | --- |
| calibration | cits | 0.044102 | 0.187163 | 85.871086 |
| calibration | synthetic_control | 0.058055 | 0.163570 | 85.871086 |
| test | cits | 0.052309 | 0.199634 | 94.428688 |
| test | synthetic_control | 0.120346 | 0.172944 | 94.428688 |
| train | cits | 0.515962 | 0.523462 | 73.884425 |
| train | synthetic_control | 0.494847 | 0.487347 | 73.884425 |
| validation | cits | 1.453943 | 0.847225 | 87.815159 |
| validation | synthetic_control | 1.588598 | 0.722018 | 87.815159 |

The selected error learner overpredicts test error on average. Different epidemic periods, historical normalization scales and area composition can affect transfer; these are possible explanations, not established causes. The table reports the actual shifts.

Largest excess errors relative to CITS:

| experiment_id | cits | dynamic | dynamic_minus_cits | model_disagreement | outcome_scale |
| --- | --- | --- | --- | --- | --- |
| niteroi_zone_v3_zone_3_201510_3 | 0.125500 | 0.350126 | 0.224626 | 0.243782 | 78.895238 |
| niteroi_zone_v3_zone_3_201507_6 | 0.069581 | 0.170214 | 0.100633 | 0.135682 | 81.107843 |
| niteroi_zone_v3_zone_3_201504_3 | 0.031432 | 0.070935 | 0.039503 | 0.098175 | 83.262626 |
| niteroi_zone_v3_zone_4_201510_3 | 0.052313 | 0.091509 | 0.039197 | 0.108708 | 81.485714 |
| niteroi_zone_v3_zone_2_201510_3 | 0.106104 | 0.127905 | 0.021801 | 0.021711 | 109.990476 |

The frozen score threshold retains 21/21 test experiments. No test error reduction from abstention is demonstrated at this operating point. The full error-coverage curve is descriptive and is not used to retune the threshold. No models or settings were changed in response to these errors.
