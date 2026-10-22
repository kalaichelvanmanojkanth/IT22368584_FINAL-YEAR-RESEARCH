# Frozen exploratory design

This design was implemented after Phase B's baseline results were inspected. It is
not preregistration. The current Phase C/D/E extension follows previous inspection
of the 2015 test, so its results are exploratory, not fresh confirmatory evidence.
Repeated future tuning on these reported results would invalidate a final-test interpretation.

- Training labels: complete pseudo windows ending by December 2012.
- Validation: windows wholly within 2013; model family, temperature and thresholds.
- Calibration: windows wholly within 2014; all areas/windows grouped by calendar
  year, leaving only one dependence block. Phase B already inspected 2014 scores.
- Test: windows wholly within 2015, excluded from this run's fitting, selection and calibration.
  Zone 1 is ineligible due to the earlier pilot releases.
- 2016 is excluded from reliability evaluation because Phase B reported its scores.
- Every area, estimator and horizon shares calendar boundaries. Crossing windows
  are purged. Overlapping windows within a split remain dependent.
- Counterfactual baselines use expanding historical information; reliability model
  parameters remain frozen when evaluating later periods.
- Scale normalization uses max(pre-period mean count, 1), never target outcomes.
- Both GradientBoostingRegressor and HistGradientBoostingRegressor have fixed
  settings (squared error, learning rate .05, depth 2, minimum leaf 4, 60 iterations,
  seed 42). Histogram boosting additionally uses at most 4 leaves, L2=1, 255 bins,
  no internal early stopping. Full actual parameters are saved in frozen_design.json.
- Both models fit training records only, with training-only median imputation,
  missing indicators and model-name encoding. All-null features are excluded.
- Select model family and temperature in [.1, .3, 1, 3] by minimum validation
  ensemble normalized MAE. Ties use family name then lower temperature.
- A non-ML expected-error comparator predicts the training mean error per estimator.
  The estimator with lowest training mean is the fixed historically-best comparator.
- Score: 1 / (1 + predicted normalized error / median training error). This is not
  clinical risk, a causal probability, or a calibrated ensemble-error expectation.
- Missing outcomes, unseen horizons/gaps trigger abstention. Single-donor analyses
  and sparse calibration require review. The minimum block count is a configurable
  evidence safeguard, not a statistical coverage guarantee.
- Choose a score threshold on validation only by minimum retained normalized MAE
  subject to at least 50% validation coverage. Candidates are 0 and observed scores;
  ties prefer greater coverage then lower threshold. Freeze before test evaluation.
  Selection uses the same validation set as model/temperature selection, which can
  overfit that small set. Test coverage and error are observed, never guaranteed.
- Error-versus-coverage curves are descriptive: do not use test curves to select
  another operating point. Score retention differs from the full evidence policy.
- This run stops at Phase C/D/E. Earlier robustness/placebo/semi-synthetic artifacts
  belong to their original runs and are not regenerated or pooled into this result.

The present hypothesis is not established. Finding a simple baseline performs
better, or abstaining on every actual intervention, is a valid research outcome.
