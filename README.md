# RACE-Dengue

Repository: [IT22368584_FINAL-YEAR-RESEARCH](https://github.com/kalaichelvanmanojkanth/IT22368584_FINAL-YEAR-RESEARCH).

For Streamlit Community Cloud, select this repository, branch **main**, app file
**app/streamlit_app.py**, and **Python 3.13** in Advanced settings. The root
`requirements.txt` includes the tested dependency lockfile. The verified public
source workbooks and saved model artifacts are included. See
[deployment instructions](docs/DEPLOYMENT.md); an online deployment still requires
verification after you create it on Streamlit.

**Reliability-Aware Counterfactual AI for Evaluating Dengue-Control Interventions
Using Multi-Site Surveillance Data** — standalone final-year research Component 03.

The Niteroi research pipeline implements data auditing, CITS and Synthetic Control,
historical pseudo-interventions, a trained supervised reliability model, dynamic
weighting, empirical uncertainty, explicit abstention, robustness experiments,
a Streamlit dashboard and an optional FastAPI interface.

**The reliability hypothesis is not established.** Current results must be read in
[the computed PP1 summary](reports/pp1_summary.md), including comparisons with simple
baselines and failed/unavailable analyses. A working implementation does not prove
an improvement. Recorded interventions can correctly receive insufficient evidence.

The current Phase C/D/E run compares GradientBoostingRegressor,
HistGradientBoostingRegressor and a non-ML historical-error predictor. Validation
selects the error model, weighting temperature and abstention threshold. On the
previously inspected 2015 test, the selected ensemble loses to CITS and slightly
to equal weights. Its frozen threshold retains every test window. See the
[computed error analysis](reports/cde_error_analysis.md); this is exploratory evidence.

## Start the dashboard

From this directory, after installing the dependencies:

```bash
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The dashboard reads actual source data and saved results. Its pages are **Overview**,
**Data Audit**, **Counterfactual Analysis**, **Model Comparison**, **Reliability**,
**Robustness**, and **Export**. Select **Counterfactual Analysis**, choose an area
and configured horizon, and run the real pipeline. Source dates are fixed to the
reviewed chronology. Unsupported estimates display **Insufficient Evidence**;
effects remain null, while clearly labeled exploratory trajectories can remain visible.

Missing or stale artifacts produce an explicit unavailable state. Changing area,
horizon or research run invalidates the old analysis. Current-run robustness is
kept separate from explicitly selected archived runs. JSON, CSV and Markdown exports
include the actual status and warnings. No screenshot/sample research scores are hardcoded.

The documented local launch command binds to loopback, and headless startup means the
optional onboarding email prompt does not block launch. Usage statistics are disabled.

Browser verification and screenshots live in `artifacts/figures/pp1_dashboard/`.
To reproduce them with the app running locally, install optional tooling with
`.venv/bin/python -m pip install playwright`, then run
`.venv/bin/python scripts/capture_pp1_dashboard.py`. The script uses an isolated
headless Chrome session; use `--channel chromium` after `playwright install chromium`
if Chrome is unavailable. It checks downloads against a fresh pipeline analysis.
Run `.venv/bin/python -m src.pipeline.dashboard_reporting` to refresh the recorded
UI status in the PP1 summary after browser verification.

## Install and reproduce

Python 3.11+ is required; the recorded local environment uses Python 3.13/macOS ARM.
Do not copy or commit `.venv` between machines.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m src.pipeline.reproduce
.venv/bin/python -m pytest -q --junitxml=artifacts/metrics/all_tests.xml
.venv/bin/python -m src.pipeline.research_reporting
```

For another Python/platform, if the exact lock is unavailable, resolve compatible
versions with `pip install -e '.[dev,ui,api]'` and retain the new run's dependency
manifest. The core requires no GPU or deep-learning framework. Configurations live
in `configs/niteroi.yaml` and `configs/experiments.yaml`.

Individual phases:

```bash
.venv/bin/python -m src.pipeline.evaluate download
.venv/bin/python -m src.pipeline.evaluate audit
.venv/bin/python -m src.pipeline.evaluate baselines
.venv/bin/python -m src.pipeline.pseudo
.venv/bin/python -m src.pipeline.train
.venv/bin/python -m src.pipeline.robustness
.venv/bin/python -m src.pipeline.predict --location zone_2 --horizon 6
.venv/bin/python -m src.pipeline.research_reporting
```

To reproduce only the current Phase C/D/E extension using the existing audited
data and unchanged baselines:

```bash
.venv/bin/python -m src.pipeline.pseudo
.venv/bin/python -m src.pipeline.train
.venv/bin/python -m pytest -q --junitxml=artifacts/metrics/all_tests.xml
.venv/bin/python -m src.pipeline.research_reporting
```

The Phase C/D/E reporter generates comparisons and error analysis without running
additional methods. Older robustness and recorded-intervention exports retain
their original provenance and are separate from this run.

Changing the dataset/config requires regenerating pseudo experiments before
training. The saved training design is frozen before test evaluation. Repeatedly
tuning against the reported test results would invalidate held-out evaluation.

## Data and design

- [Original Niteroi publication](https://doi.org/10.1371/journal.pntd.0009556)
- [Versioned original dataset](https://doi.org/10.6084/m9.figshare.13662203.v3), CC BY 4.0
- [Manual download/checksum instructions](docs/MANUAL_DOWNLOAD.md)
- [Exact data dictionary](docs/DATA_DICTIONARY.md)
- [Frozen exploratory design](docs/EXPERIMENT_DESIGN.md)
- [Computed audit](reports/dataset_audit.md) and [methodology](reports/methodology.md)

Raw files are checksum verified and never overwritten. Missing data remain missing;
failed downloads never become random observations. Synthetic fixtures exist only
in tests; semi-synthetic experiments copy real historical trajectories and explicitly
label injected effects. No other group component is required.

Clean history ends in 2014 for Zone 1 and 2016 for Zones 2–4 to exclude pilot/staged
releases. Source monitoring dates define the actual post-analysis windows. The
pre-fitted CITS variant is not a reproduction of the publication's full regression.
The sole permanent control makes real-data SCM a single-donor comparator.

The reliability learner predicts historical held-out normalized MAE. Calendar-based
train/validation/calibration/test partitions prevent random row leakage; crossing
windows are purged. Models reject post-treated outcomes in training and prediction
inputs. Contemporaneous control outcomes are permitted for retrospective analysis,
not prospective outbreak forecasts.

## Evidence and uncertainty

Model Reliability Score is a relative expected-error score, **not a probability of
causality or clinical risk**. Evidence status is one of `SUPPORTED_ESTIMATE`,
`UNCERTAIN_REVIEW_REQUIRED`, or `INSUFFICIENT_EVIDENCE`. Main effect fields are null
unless support criteria are met. Labelled exploratory model contrasts remain
inspectable; they must not be treated as supported intervention conclusions.

Uncertainty uses empirical residual maxima grouped across whole calibration years.
The current dataset provides only one such year: exact coverage is not claimed.
Missingness stress triggers explicit abstention rather than fabricated imputation.
In-space and donor-removal checks report unavailable when no valid donor remains.

## Outputs

- `reports/pp1_summary.md`: completed work, actual comparisons, limitations and saved test counts.
- `reports/results_table.csv`: per-window strategy evaluation; `baseline_results.csv`: Phase B outputs.
- `reports/intervention_evidence.csv`: recorded intervention statuses and reasons.
- `artifacts/latest_pseudo.json`, `latest_reliability.json`: links to versioned run directories.
- `artifacts/runs/<run_id>/`: trained model/bundle, frozen design, source hash, config,
  dependencies, predictions, diagnostics, comparisons, failures and robustness tables.
- `artifacts/predictions/`: recorded-intervention JSON plus baseline CSV trajectories.
- `artifacts/figures/<run_id>/`: high-resolution plots covering all requested figure categories.

Reports are regenerated from saved data; run the reporting command after tests to
refresh test counts. The application exposes saved failures and warnings. Notebooks
read reusable module outputs and contain no fabricated cached results.

## Optional API and component boundaries

```bash
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

See [API contract and example](docs/INTEGRATION.md). C02 may pass location/context;
C04 may consume an evidence object. Neither is mandatory.

| Component | Question |
|---|---|
| C01 | What is the dengue risk / what may happen? |
| C02 | Which community requires attention first? |
| C03 | Did a recorded intervention make a measurable difference, and how reliable is that conclusion? |
| C04 | How should available resources be allocated? |

The structural/state-space baseline remains explicitly pending behind its common
interface; it never returns fake output. Separate source adapters and audits are implemented for Yogyakarta and Colombia.
Run `.venv/bin/python -m src.data.external`; see [external comparability review](docs/EXTERNAL_SITES.md).
Different case definitions, contamination/staged exposure and unverified permanent
control sets preclude pooling or use by the Niteroi estimator without a revised
design. There are no external-site performance claims.

## Limitations

- Observational/quasi-experimental estimates rely on assumptions.
- Imperfect controls can bias effects; spillover may contaminate controls.
- Surveillance/reporting changes can affect results.
- Public international results are context-specific.
- Wolbachia estimates must not automatically be interpreted as effects of fogging/source reduction.
- International estimates must not automatically be transferred to Sri Lanka.
- Low-reliability analyses should receive human review.
- This system does not diagnose or treat patients.

See [full limitations](reports/limitations.md). This repository implements no
clinical diagnosis, general outbreak prediction, community risk ranking or resource
allocation. Its present findings do not prove the reliability-learning hypothesis.
