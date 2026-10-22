# Methodology — baseline specification and reliability extensions

Research question: Can a trained reliability-aware counterfactual framework improve
the dependability of dengue intervention estimates and abstain when evidence is
insufficient? A reliability learner is now trained; results are exploratory and do not establish the hypothesis.

## Current Phase C/D/E extension

The current run compares two training-only expected-error learners:
GradientBoostingRegressor and HistGradientBoostingRegressor, plus a training-mean
non-ML error baseline. Validation jointly selects the learner and softmax temperature;
it then selects a score threshold minimizing retained normalized MAE subject to
at least 50% validation coverage. Settings are frozen before scoring test windows.
The threshold is also applied by the existing evidence policy. Score-based coverage
is distinct from a supported causal conclusion.

See [the current design](../docs/EXPERIMENT_DESIGN.md) and
[generated PP1 results](pp1_summary.md) for exact parameters, features, split sizes,
comparisons, abstention behavior and error analysis. The previously inspected 2015
test makes this extension exploratory. No methods beyond Phase C/D/E are added.
The later C–H section describes earlier work; its single-learner selection description
is superseded by the current design, and its robustness artifacts belong to earlier runs.

## Design and provenance

The MVP uses the original Niteroi zone-level workbook, Figshare
10.6084/m9.figshare.13662203.v3, CC BY 4.0. Original publication:
https://doi.org/10.1371/journal.pntd.0009556. Workbook Data/Metadata sheets define
the source variables; docs/DATA_DICTIONARY.md documents the exact mapping.
Unknown exposure and coordinates stay null. Months are repeated observations
within geographic units, not independent interventions.

Source release and monitoring dates differ. Fit history ends December 2014 in
Zone 1 (pilot releases occurred in 2015–16) and December 2016 in Zones 2–4.
These deliberately conservative cutoffs avoid treating the interval before
zone-wide rollout as guaranteed unexposed. Six post months start at the source
monitoring date. Long excluded gaps increase extrapolation risk. Defaults are
chosen before observing validation errors; they are not claimed optimal.

## CITS counterfactual baseline

This is a pre-fitted, control-adjusted interrupted-time-series counterfactual,
not a reproduction of the paper's segmented treatment-interaction regression.
For treated counts Y_t, the model is:

    Y_t ~ NegativeBinomial(mu_t, alpha)
    log(mu_t) = log(population_t) + beta_0 + beta_1 * years_t
                + sum_h [a_h sin(2 pi h month_t/12) + b_h cos(2 pi h month_t/12)]
                + sum_j gamma_j log(1 + control_incidence_jt)

Two Fourier harmonics and alpha=1 are configurable, prespecified defaults.
Poisson is an optional family. No post-treatment outcome is used to fit a
coefficient. The intervention indicator is fixed at zero for Y(0); a treatment
coefficient cannot be identified from an all-pre-treatment fitting sample.
The interruption defines the later contrast window. A conventional full-period
treatment regression would be a separate design and is not silently substituted.
Population is an offset. Control incidence is cases per 100,000.
Residuals, pre MAE/RMSE, lag-one autocorrelation, design condition and parameters
are saved. Negative-binomial dispersion is not empirically tuned in Phase B.

## Synthetic Control

Minimize mean((Y1 - Y0 w)^2) with w_j >= 0 and sum(w_j)=1 using SLSQP.
Both sides are divided by one common pre-period scale for numerical stability;
this does not change the minimizer. Default matching is incidence per 100,000,
then predictions are converted to counts using the treated population.
Raw-count matching is configurable and requires comparable population scales.
Optimisation failure is fatal for that analysis and is saved in the results table.
The sole permanent untreated zone makes the real-data weight exactly one: this
is the only feasible simplex solution. It does not establish donor diversity or
learned weighting benefits. Other treated zones are not repurposed as controls.
Maximum/effective donor counts, residuals and matching error are saved.
Leave-one-donor-out is not possible for this one-donor real-data specification.

## Leakage prevention and baseline evaluation

Models reject fit rows at/after the analysis interruption, source-exposed rows,
rows beyond the declared clean cutoff, mixed studies, and invalid donors.
Prediction accepts only post control outcomes plus a separate treated population
covariate. Treated outcomes are accessed after prediction for scoring/contrasts.
Contemporaneous control outcomes are allowed for retrospective causal estimation
under no spillover; this is not prospective outbreak prediction.

A fixed last-12-month historical holdout within each clean period assesses each
baseline. It is not the Phase C pseudo-experiment training engine. No tuning is
performed on these holdouts; future tuning must use different validation data.
Shared control/calendar windows make these scores dependent; no independence-based
significance claim is made. MAE, RMSE and seasonal MASE (12-month pre-series scale)
are reported per location/model. MASE is null when its scale is zero or undefined.
Actual post-intervention factual-versus-counterfactual differences are not scored
as prediction errors because the untreated outcome is unobserved.

## Effect contrasts and uncertainty

    effect_absolute = sum_t [observed_t - counterfactual_t]
    estimated_cases_averted = -effect_absolute
    effect_percent = 100 * effect_absolute / sum_t counterfactual_t

Negative absolute/percent contrast denotes fewer observed notified cases than
estimated without intervention; cases-averted can be negative and is not clipped.
Relative contrast is null if the denominator is <= 1e-8. These are observational
model contrasts, not guaranteed causal effects or counts of confirmed infections.
No temporal calibration or bootstrap uncertainty is implemented in Phase B.
All interval and reliability fields remain null. Every successful analysis is
marked UNCERTAIN_REVIEW_REQUIRED because validated reliability is unavailable;
this is a phase-status safeguard, not validation-derived selective prediction.
Failure rows remain explicit. No support threshold or probability is fabricated.


The preceding interval/status description refers to native Phase B baseline outputs. The current ensemble calibration and evidence policy are specified below.



The preceding interval/status description refers to native Phase B baseline outputs. The current ensemble calibration and evidence policy are specified below.



The preceding interval/status description refers to native Phase B baseline outputs. The current ensemble calibration and evidence policy are specified below.


## Phases C–H: historical reliability experiments

The supervised target is the estimator's actual pseudo-post MAE divided by the
mean pre-period count (minimum 1). Features are computed before target outcomes
are read. Allowed contemporaneous post-control outcomes contribute only to
retrospective prediction/disagreement/extrapolation. No post treated outcome is a
feature. All estimator/area records use the same calendar partitions: training
through 2012, validation 2013, calibration 2014, final evaluation 2015. Windows
crossing boundaries are purged. Source clean-history cutoffs still apply, so Zone 1
has no 2015 test. Horizons are 3 and 6 months, spaced quarterly. Overlapping windows
and shared controls are dependent; area-month rows are not independent trials.
Prior Phase B inspected historical 2014/2016 scores. This study is exploratory and
not a preregistered untouched confirmatory study. No model selection uses 2015
labels. Training settings and temperature are frozen before evaluating that split.

A GradientBoostingRegressor with fixed squared-error loss, 60 trees, depth 2,
minimum leaf size 4 and learning rate .05 learns normalized expected error.
Numeric median imputation and estimator encoding are fitted on training only.
Entirely unavailable features are excluded and listed in the manifest. Scale
normalization is known from each historical pre-window. This model is separate
from C01, C02 and C04 and does not require their outputs.

Dynamic weights are softmax(-predicted_error / temperature). Temperature is chosen
from the config grid by validation-window ensemble normalized MAE. Comparison
strategies include each estimator, equal weights and the estimator with the lowest
TRAINING historical average normalized error. No weights are tuned using test
outcomes. Model Reliability Score = 1/(1 + weighted predicted normalized error /
median training normalized error). This is a relative error score, not a probability
of causality or clinical risk, and not a calibrated ensemble-error expectation.

The evidence policy compares diagnostic values with validation-derived quantiles.
Unsupported horizons, missing outcomes and training-to-intervention gaps outside
validation support trigger insufficient evidence. Sparse calibration and a single
donor require review. The minimum calibration-block count is a configurable
prespecified evidence safeguard, not a validated guarantee. Main effect fields are
null when evidence is not supported; explicitly labelled exploratory contrasts
remain inspectable and exportable. No automatic real-world decision is prescribed.

Empirical uncertainty uses the maximum absolute scaled ensemble residual across
all calibration windows, dates and locations in each calendar YEAR. Keeping entire
years together avoids pretending repeated correlated months are independent.
A high empirical quantile of these year maxima gives trajectory bounds, clipped
at zero below; aggregate effect bounds use observed totals minus trajectory-bound
totals. This is an empirical diagnostic interval, not an exact conformal guarantee.
Only one calibration year is available here: the empirical quantile is therefore
just its maximum and provides weak evidence. Test coverage is measured and reported,
not promised. Reliable inference requires more independent calibration periods/sites.

In-time placebos report apparent signed effects for historical clean windows.
In-space and donor-removal implementations require valid remaining untreated donors;
the real Niteroi zone design cannot supply them. Artificial stress scenarios retain
failures for missingness, insufficient history and removal of the sole donor.
Missing outcomes are not imputed to force predictions. Control-reporting inflation
is a labelled numerical contamination scenario, not observed biological spillover.
All stress evaluations use the frozen model and do not cause reselection.

Semi-synthetic experiments multiply held-out REAL clean outcomes by (1-reduction)
in a copy. The untreated original trajectory is known; predictions are generated
without injected targets. Bias, absolute/relative error, aggregate-effect interval
coverage and width are computed for reductions 0, .25 and .5. Fractional injected
trajectories are methodological constructs, never new real case observations.
The same forecasting error reappears under additive contrasts across effect sizes;
these injections do not create independent information or prove real-world causality.

Ablations remove diagnostics with the full model's fixed temperature. Unavailable
or constant diagnostic groups are marked not identifiable; no contribution is
claimed for them. Equal weighting and best-single comparisons are in the main
strategy table. External sites require reviewed mappings and study-specific clean
windows; no blind pooling or external performance claims are made. The optional
local FastAPI and Streamlit interfaces use the same standalone analysis function.
