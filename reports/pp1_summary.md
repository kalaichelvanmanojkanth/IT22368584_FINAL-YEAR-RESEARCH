# PP1 summary — computed Phase C/D/E results

Run: `20261021T215308Z_52d8d5dc`; pseudo source: `20261021T214953Z_a2b8e4eb`.

The proposed ensemble does not beat all required baselines on the held-out mean normalized MAE. The improvement hypothesis is not established.

Data audit: 810 monthly rows and 5 Niteroi zones. Original source and audit are documented in `dataset_audit.md`. Existing CITS/SCM implementations and their baseline outputs were inspected and preserved.

## Experiments and separation

Valid pseudo-intervention experiments: **169**; estimator records: **338**; fitting failures: **0**.

| split | experiments | model_rows | calendar_origins | first_target | last_target |
| --- | --- | --- | --- | --- | --- |
| train | 92 | 184 | 12 | 2010-01-01 | 2012-12-01 |
| validation | 28 | 56 | 4 | 2013-01-01 | 2013-12-01 |
| calibration | 28 | 56 | 4 | 2014-01-01 | 2014-12-01 |
| test | 21 | 42 | 4 | 2015-01-01 | 2015-12-01 |

Each experiment is one area/start/horizon, with paired CITS and SCM records. Minimum history is 36 months; starts are quarterly; horizons are 3 or 6 months. The entire outcome window must precede the real intervention clean cutoff (Zone 1: December 2014; others: December 2016). Windows crossing split boundaries are purged. Related areas, horizons and estimators share calendar partitions. Training labels alone fit preprocessing and error models. Validation selects model, temperature and threshold. Calibration is reserved for existing interval calibration. Counterfactual baselines use expanding histories available at each pseudo date; meta-model parameters stay frozen. This is rolling retrospective evaluation, not a single-origin future forecast.

## Features and expected error

Numeric features: `pre_rmse`, `pre_mae`, `trend_mismatch`, `residual_variance`, `residual_autocorrelation`, `seasonal_residual_strength`, `missing_rate`, `horizon`, `max_donor_weight`, `effective_donor_count`, `model_disagreement`, `extrapolation_features`, `available_history`, `gap_months`.

Categorical feature: `model_name` (one-hot encoded). Median imputation and missing indicators are fitted on training only. Omitted because entirely unavailable: leave_one_donor_sensitivity, placebo_error. Constant among measured training values: missing_rate, max_donor_weight, effective_donor_count, gap_months. Donor diagnostics are missing for CITS and constant for the one-donor SCM. Outcome scale is used to define the target and convert predictions to cases; it is not a learned input.

Target: pseudo-post MAE / max(pre-period mean cases, 1). Features exclude target outcomes, errors, experiment IDs and split labels. Contemporaneous untreated-control observations enter retrospective prediction/disagreement/extrapolation. Individual expected errors are exported in normalized units and cases. The weighted predicted estimator error is a selection proxy; it is not a separately calibrated expectation of ensemble error.

## Models and validation decisions

Selected family: **gradient_boosting**; softmax temperature: **3.0**.

Both families use squared-error loss, learning rate 0.05, depth 2, minimum leaf size 4 and seed 42. Gradient boosting uses 60 trees and subsample 1. Histogram gradient boosting uses 60 iterations, 4 maximum leaves, L2 regularization 1, 255 bins and disabled internal early stopping. Full actual estimator parameters:

```json
{
  "gradient_boosting": {
    "alpha": 0.9,
    "ccp_alpha": 0.0,
    "criterion": "deprecated",
    "init": null,
    "learning_rate": 0.05,
    "loss": "squared_error",
    "max_depth": 2,
    "max_features": null,
    "max_leaf_nodes": null,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 4,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 60,
    "n_iter_no_change": null,
    "random_state": 42,
    "subsample": 1.0,
    "tol": 0.0001,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": false
  },
  "hist_gradient_boosting": {
    "categorical_features": null,
    "early_stopping": false,
    "interaction_cst": null,
    "l2_regularization": 1.0,
    "learning_rate": 0.05,
    "loss": "squared_error",
    "max_bins": 255,
    "max_depth": 2,
    "max_features": 1.0,
    "max_iter": 60,
    "max_leaf_nodes": 4,
    "min_samples_leaf": 4,
    "monotonic_cst": null,
    "n_iter_no_change": 10,
    "quantile": null,
    "random_state": 42,
    "scoring": "loss",
    "tol": 1e-07,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": false
  }
}
```

| model_kind | temperature | validation_normalized_mae |
| --- | --- | --- |
| gradient_boosting | 0.100000 | 1.626340 |
| gradient_boosting | 0.300000 | 1.563994 |
| gradient_boosting | 1.000000 | 1.479921 |
| gradient_boosting | 3.000000 | 1.466722 |
| hist_gradient_boosting | 0.100000 | 1.613679 |
| hist_gradient_boosting | 0.300000 | 1.540905 |
| hist_gradient_boosting | 1.000000 | 1.479671 |
| hist_gradient_boosting | 3.000000 | 1.468268 |

Non-ML expected-error baseline: training mean normalized error for each estimator. Training averages: {'cits': 0.5159618753444715, 'synthetic_control': 0.49484717260508976}. Fixed historically-best estimator: **synthetic_control**.

Weights are softmax(-predicted normalized error / temperature), recalculated for each experiment. The reliability score is 1 / (1 + weighted predicted error / median training target error). Model family and temperature minimize mean validation ensemble normalized MAE. The score threshold minimizes validation retained error subject to at least 50% validation coverage; ties prefer greater coverage. The threshold is frozen before test evaluation and applied using scores alone.

Threshold: **0.215104414004**; validation retained **53.571429%**; retained validation error **0.342055**.

## Actual test comparisons

| strategy | mae | rmse | normalized_mae |
| --- | --- | --- | --- |
| cits | 5.020993 | 6.146944 | 0.052309 |
| synthetic_control | 10.800339 | 13.570745 | 0.120346 |
| equal_weight | 6.919152 | 8.886823 | 0.074462 |
| best_historical | 10.800339 | 13.570745 | 0.120346 |
| dynamic | 6.944318 | 8.917592 | 0.074766 |

Lower is better. MAE and RMSE are in monthly notified cases. Each entry is the mean of per-experiment metrics (RMSE is not pooled). Normalized MAE divides each experiment by its own pre-period mean count, floored at 1. All five methods use the same 21 windows.

By area (normalized MAE):

| treated_location | cits | synthetic_control | equal_weight | best_historical | dynamic |
| --- | --- | --- | --- | --- | --- |
| zone_2 | 0.063346 | 0.085480 | 0.074182 | 0.085480 | 0.074224 |
| zone_3 | 0.054634 | 0.199591 | 0.099406 | 0.199591 | 0.100275 |
| zone_4 | 0.038946 | 0.075968 | 0.049798 | 0.075968 | 0.049798 |

By horizon (normalized MAE):

| horizon | cits | synthetic_control | equal_weight | best_historical | dynamic |
| --- | --- | --- | --- | --- | --- |
| 3 | 0.054665 | 0.129539 | 0.080127 | 0.129539 | 0.080452 |
| 6 | 0.049167 | 0.108089 | 0.066909 | 0.108089 | 0.067184 |

Both candidate ensembles, each using its own validation-selected temperature:

| model_kind | temperature | mae | rmse | normalized_mae |
| --- | --- | --- | --- | --- |
| gradient_boosting | 3.000000 | 6.944318 | 8.917592 | 0.074766 |
| hist_gradient_boosting | 3.000000 | 6.932380 | 8.902932 | 0.074625 |

Histogram results are reported for comparison; they do not change the validation-selected family.

## Expected-error prediction and selective evaluation

| error_predictor | error_prediction_mae | error_prediction_rmse | mean_predicted_error | mean_actual_error |
| --- | --- | --- | --- | --- |
| gradient_boosting | 0.126052 | 0.140831 | 0.186289 | 0.086328 |
| hist_gradient_boosting | 0.124763 | 0.136865 | 0.185744 | 0.086328 |
| historical_model_average | 0.423681 | 0.430833 | 0.505405 | 0.086328 |

Expected-error prediction metrics pool the 42 test estimator records, in normalized-error units. These differ from counterfactual case-prediction accuracy. Per-area rank correlations and high-error AUC are in `meta_metrics.csv`; AUC is unavailable when the test subset lacks both classes.

| split | location | threshold | analyses | retained_count | coverage | all_error | retained_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | ALL | 0.215104 | 21 | 21 | 1.000000 | 0.074766 | 0.074766 |
| validation | ALL | 0.215104 | 28 | 15 | 0.535714 | 1.466722 | 0.342055 |

Score-based retained coverage is separate from the full evidence policy: diagnostic support checks, single-donor warnings and sparse calibration still apply. Retaining a pseudo experiment does not authorize a supported causal estimate. See `cde_error_analysis.md` for paired errors and transfer analysis.

## Verification

```json
{
  "source": "artifacts/metrics/all_tests.xml",
  "tests": 81,
  "failures": 0,
  "errors": 0,
  "skipped": 0,
  "passed": 81
}
```

Tests cover feature/target separation, pre-intervention eligibility, calendar separation, both learners, training-only preprocessing, validation-only threshold selection, frozen inference, artifact persistence and existing application behavior.

## Methodological limitations

- This 2015 test was already inspected in earlier work. This extension is exploratory; computational split separation does not make it a fresh confirmatory study.
- There are only four test calendar origins, overlapping 3/6-month windows, shared controls and shared expanding histories. The 21 experiments are dependent; no independent-sample significance claim is made.
- Only one permanent untreated donor is available. SCM has no donor-choice flexibility; donor-removal sensitivity and in-space checks are unavailable. Zone 1 has no test experiments.
- Validation is one epidemic year. Selecting family, temperature and threshold on the same small validation set can overfit it; the coverage floor is a design setting, not a guarantee of test coverage or risk.
- Missingness and several support diagnostics do not vary in these complete-data experiments. They cannot establish robustness or feature contributions.
- Contemporaneous donor data make this retrospective counterfactual evaluation, not operational forecasting. Seasonal shifts, spillover and unmeasured confounding remain possible.
- Pseudo-post prediction error is observable; real intervention counterfactual error and causal effect are not. Good pseudo prediction would not itself establish a causal claim.
- Existing empirical interval calibration contains 1 calendar year; it has no exact conformal or causal coverage guarantee. This extension does not add uncertainty methods.
- No external-site generalization, structural time-series extension, new robustness methods or post-test tuning is included in this run. Previous phase artifacts remain separate.

## Generated artifacts

- `reports/results_table.csv`: per-experiment comparisons, split labels and frozen selection decisions.
- `reports/test_comparison.csv`: test mean case errors and normalized errors.
- `reports/expected_error_comparison.csv`: error-prediction quality for both learners and the non-ML baseline.
- `reports/cde_error_analysis.md`: actual paired failures and distribution comparisons.
- `artifacts/runs/20261021T215308Z_52d8d5dc/`: frozen design, both models, non-ML baseline, diagnostics, predictions and evaluation tables.
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_test_comparison.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_test_by_zone.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_validation_selection.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_expected_error.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_dynamic_weights.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_error_coverage.png`
- `artifacts/figures/20261021T215308Z_52d8d5dc/cde_paired_errors.png`

Reproduce with `python -m src.pipeline.pseudo`, `python -m src.pipeline.train`, `python -m pytest -q --junitxml=artifacts/metrics/all_tests.xml`, then `python -m src.pipeline.research_reporting`. Work stops at Phase C/D/E.

## PP1 demonstration interface

Browser verification recorded: 2026-10-22T04:33:22.108825+00:00.
Verified research run: `20261021T215308Z_52d8d5dc`.
Recorded browser outcome: **passed**. Source files and active research run still match verification: **True**.

Implemented and exercised in the running Streamlit application:

- Overview: research problem, question, component boundary and artifact-derived comparisons.
- Data Audit: original metadata, observations, locations, period, missingness and intervention chronology; pipeline validation is recomputed.
- Counterfactual Analysis: source study/area, configured horizons, locked source dates and real observed/CITS/SCM/ensemble trajectories.
- Model Comparison: paired experiment metrics, validation diagnostics and learned weights from the current run.
- Reliability: actual score, prominent evidence status, measured factors, warnings and the frozen error-versus-coverage operating point.
- Robustness: current-run availability and explicitly selected archived placebo, donor and stress artifacts. Older results are never presented as current-run validation.
- Export: actual analysis JSON, status-bearing CSV and Markdown report; unsupported effects remain null.
- The requested About/Limitations text is displayed on every page.

Browser pages checked: Overview, Data Audit, Counterfactual Analysis, Model Comparison, Reliability, Robustness, Export.
Download contents compared with fresh pipeline analysis: **True**.
Uncaught browser errors: **0**.

The real recorded-intervention demonstration returned `INSUFFICIENT_EVIDENCE`; it displays warnings and exploratory trajectories while withholding a supported effect. No research models were retrained or methods added for the UI.

Recorded dashboard tests: `{"tests": 7, "failures": 0, "errors": 0, "skipped": 0, "passed": 7}`.
These include artifact-to-UI equality, source/configuration mismatch, missing artifacts, stale selection clearing, archived-run labeling and null-preserving abstention exports.

Saved browser screenshots: **9**.
- `artifacts/figures/pp1_dashboard/01_overview.png`
- `artifacts/figures/pp1_dashboard/02_data_audit.png`
- `artifacts/figures/pp1_dashboard/03_counterfactual_insufficient_evidence.png`
- `artifacts/figures/pp1_dashboard/04_model_comparison.png`
- `artifacts/figures/pp1_dashboard/05_reliability.png`
- `artifacts/figures/pp1_dashboard/06_robustness_current.png`
- `artifacts/figures/pp1_dashboard/06b_robustness_archived.png`
- `artifacts/figures/pp1_dashboard/07_export.png`
- `artifacts/figures/pp1_dashboard/08_mobile_overview.png`

Verification record: `artifacts/figures/pp1_dashboard/verification.json`.
Launch locally: `.venv/bin/python -m streamlit run app/streamlit_app.py`.
Recheck the running UI: `.venv/bin/python scripts/capture_pp1_dashboard.py`.
The browser tooling is optional; Streamlit does not require Playwright to run.
