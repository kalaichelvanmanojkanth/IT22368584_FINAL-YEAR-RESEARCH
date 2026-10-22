"""Generate honest Phase A/B research reports from saved run artifacts."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

METHODOLOGY = """# Methodology — implemented Phases A/B

Research question: Can a trained reliability-aware counterfactual framework improve
the dependability of dengue intervention estimates and abstain when evidence is
insufficient? This hypothesis is untested. No reliability learner is trained yet.

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

## Reproduction and deferred work

Each run records UTC timestamp, seed, configuration, dataset SHA256, source-code
hash, dependency versions, Git commit when available, per-model parameters,
warnings, failures, predictions, residuals, metrics and 300-dpi figures.
Raw files are immutable; processed data can be regenerated.
Structural time series, pseudo-experiment generation, reliability training,
dynamic ensembles, temporal uncertainty calibration, placebo/stress experiments,
semi-synthetic recovery, cross-site adapters, dashboard and API remain pending.
The later learner should use purged chronological groups and frozen external test
sites; no cross-site pooling is justified by the current implementation.
"""

LIMITATIONS = """# Limitations and interpretation

- Observational/quasi-experimental estimates rely on assumptions; they do not prove causation.
- Imperfect controls can bias effects; spillover may contaminate controls.
- Surveillance/reporting changes can affect results and comparability over time.
- Public international results are context-specific.
- Wolbachia estimates must not automatically be interpreted as effects of fogging/source reduction.
- International estimates must not automatically be transferred to Sri Lanka.
- Low-reliability analyses should receive human review.
- This system does not diagnose or treat patients.
- One permanent control zone makes real-data SCM a single-donor comparator.
- Census denominators change in 2010; incidence and population offsets address scale,
  but not every change in population, surveillance or diagnostic practices.
- Conservative clean-history exclusions create long extrapolation gaps, especially
  for Zone 1 and Zone 4; baseline fit alone does not validate counterfactual accuracy.
- A fixed alpha negative-binomial GLM is a transparent starting model, not evidence
  that the chosen dispersion/seasonality specification is adequate.
- Holdout windows share controls and calendar periods; there are no independent
  area-month intervention replications or uncertainty claims from their sample count.
- CITS here uses a pre-fitted control regression, not the publication's full model.
- No calibrated uncertainty, trained reliability model, selective-prediction results,
  placebo validation or intervention-effect ground truth is currently available.
- Only baseline/audit figures are generated; later-phase figures are intentionally absent.
- Yogyakarta and Colombia adapters and design-comparability review are not implemented.
"""


def test_summary(root: Path) -> dict:
    """Prefer the final suite XML; never infer test success from implementation."""
    candidates = [root / "artifacts/metrics/all_tests.xml", root / "artifacts/metrics/phase_b_unit_tests.xml",
                  root / "artifacts/metrics/phase_a_tests.xml"]
    for path in candidates:
        if path.exists():
            tree = ET.parse(path).getroot()
            suites = list(tree.iter("testsuite"))
            counts = {k: sum(int(s.get(k, "0")) for s in suites) for k in ["tests", "failures", "errors", "skipped"]}
            counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
            return {"source": str(path.relative_to(root)), **counts}
    return {"source": None, "status": "No pytest XML available"}


def generate_reports(root: Path, manifest: dict, metrics: pd.DataFrame) -> None:
    directory = root / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "methodology.md").write_text(METHODOLOGY)
    (directory / "limitations.md").write_text(LIMITATIONS)
    tests = test_summary(root)
    lines = ["# PP1 summary — completed work only", "", f"Run: {manifest['run_id']}",
             f"Status: {manifest['status']}", "",
             "Research question: Can reliability-aware counterfactual AI improve dependability and abstain when evidence is insufficient?",
             "The reliability hypothesis has not been tested.", "",
             "Implemented: source pipeline, typed configuration, data validation, pre-fitted CITS and constrained Synthetic Control, leakage guards, historical baseline holdout, baseline exports/figures.",
             "Current dataset: original Niteroi zone-level Figshare v3; no synthetic research observations.",
             f"Completed baseline analyses: {manifest['completed_analyses']}; failed: {manifest['failed_analyses']}.",
             f"Phase C pseudo experiments completed: {manifest['pseudo_experiments_completed']}.",
             "Reliability-model metrics: unavailable (not trained).",
             "Robustness experiments completed: none (pending Phase F).", "",
             "## Historical holdout metrics (actual computed values)", "",
             "| Location | Estimator | MAE | RMSE | MASE |", "|---|---|---:|---:|---:|"]
    for row in metrics.to_dict("records"):
        if row["analysis"] == "clean_pre_holdout" and row["status"] == "completed":
            mase = "null" if pd.isna(row["holdout_mase"]) else f"{row['holdout_mase']:.4f}"
            lines.append(f"| {row['location_id']} | {row['model_name']} | {row['holdout_mae']:.4f} | {row['holdout_rmse']:.4f} | {mase} |")
    lines += ["", "Each holdout is a prespecified historical window, not a replicated causal-effect test.",
              "Intervention contrasts are in results_table.csv; no effect confidence intervals are claimed.",
              "", "## Tests", "", json.dumps(tests, indent=2),
              "", "Counts refer to the saved pytest XML above; rerun tests after source changes.",
              "", "## Artifacts", "", f"- {manifest['metrics_csv']}", f"- {manifest['predictions_csv']}",
              f"- artifacts/runs/{manifest['run_id']}/manifest.json"]
    lines += [f"- {path}" for path in manifest["figure_paths"]]
    failed = metrics[metrics.status.eq("failed")]
    lines += ["", "## Failed analyses", ""] + (failed.to_string(index=False).splitlines() if not failed.empty else ["None in this run."])
    lines += ["", "## Current limitations and next tasks", "",
              "Single control donor; pilot/staged-release exclusions; long extrapolation gaps; no calibrated intervals or validated reliability. See limitations.md.",
              "Stop after Phase A/B review. Next approved implementation phase: chronological pseudo-experiment engine and diagnostic feature table, followed by supervised reliability learning."]
    (directory / "pp1_summary.md").write_text("\n".join(lines) + "\n")
