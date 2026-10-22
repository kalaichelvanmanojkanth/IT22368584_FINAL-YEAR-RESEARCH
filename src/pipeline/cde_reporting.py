"""Phase C/D/E tables and figures from saved computations, without new experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.figures import plt, save_figure
from src.pipeline.reporting import test_summary
from src.utils.reproducibility import write_json

STRATEGIES = ['cits', 'synthetic_control', 'equal_weight', 'best_historical', 'dynamic']
LABELS = {'cits': 'CITS', 'synthetic_control': 'SCM', 'equal_weight': 'Equal weights',
          'best_historical': 'Historically best', 'dynamic': 'Reliability weighted'}


def markdown(table: pd.DataFrame) -> str:
    """Keep report generation independent of optional tabulate packages."""
    def cell(value):
        if pd.isna(value):
            return 'unavailable'
        if isinstance(value, (float, np.floating)):
            return f'{value:.6f}'
        return str(value).replace('|', '/')
    return '\n'.join(['| ' + ' | '.join(map(str, table.columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(table.columns)) + ' |',
                      *['| ' + ' | '.join(map(cell, row)) + ' |'
                        for row in table.itertuples(index=False, name=None)]])


def generate(root: Path) -> dict:
    manifest = json.loads((root/'artifacts/latest_reliability.json').read_text())
    directory = root/manifest['directory']
    design = json.loads((directory/'frozen_design.json').read_text())
    summary = pd.read_csv(directory/'strategy_results.csv')
    expected = pd.read_csv(directory/'expected_errors.csv')
    selective = pd.read_csv(directory/'selective_evaluation.csv')
    candidates = pd.read_csv(directory/'candidate_test_results.csv')
    tuning = pd.read_csv(directory/'temperature_validation.csv')
    risk = pd.read_csv(directory/'risk_coverage.csv')
    paired = pd.read_csv(directory/'paired_error_analysis.csv')
    # Use this reliability run's recorded source, rather than a possibly newer latest pointer.
    pseudo = json.loads((root/'artifacts/runs'/manifest['pseudo_run']/'manifest.json').read_text())
    audit = json.loads((root/'data/processed/validation.json').read_text())['statistics']
    tests = test_summary(root)
    reports = root/'reports'
    reports.mkdir(exist_ok=True)
    summary.to_csv(reports/'results_table.csv', index=False)
    test = summary[summary.split.eq('test')]
    dynamic = test[test.strategy.eq('dynamic')].copy()
    test_expected = expected[expected.split.eq('test')]
    comparison = test.groupby('strategy')[['mae', 'rmse', 'normalized_mae']].mean().reindex(STRATEGIES)
    comparison.to_csv(reports/'test_comparison.csv')
    per_area = test.groupby(['treated_location', 'strategy']).normalized_mae.mean().unstack().reindex(columns=STRATEGIES)
    per_horizon = test.groupby(['horizon', 'strategy']).normalized_mae.mean().unstack().reindex(columns=STRATEGIES)
    splits = expected.groupby('split').agg(experiments=('experiment_id', 'nunique'),
        model_rows=('experiment_id', 'size'), calendar_origins=('pseudo_date', 'nunique'),
        first_target=('pseudo_date', 'min'), last_target=('evaluation_end', 'max')).reindex(
            ['train', 'validation', 'calibration', 'test'])
    error_quality = []
    for kind, column in [('gradient_boosting', 'gradient_boosting_error_prediction'),
                         ('hist_gradient_boosting', 'hist_gradient_boosting_error_prediction'),
                         ('historical_model_average', 'historical_error_prediction')]:
        delta = test_expected[column] - test_expected.target_error
        error_quality.append({'error_predictor': kind, 'error_prediction_mae': delta.abs().mean(),
                              'error_prediction_rmse': np.sqrt(np.mean(delta**2)),
                              'mean_predicted_error': test_expected[column].mean(),
                              'mean_actual_error': test_expected.target_error.mean()})
    error_quality = pd.DataFrame(error_quality)
    error_quality.to_csv(reports/'expected_error_comparison.csv', index=False)
    candidate_comparison = candidates.groupby('model_kind').agg(temperature=('temperature','first'),
        mae=('mae','mean'), rmse=('rmse','mean'), normalized_mae=('normalized_mae','mean'))
    wins = []
    for name in STRATEGIES[:-1]:
        delta = paired['dynamic_minus_' + name]
        wins.append({'baseline': name, 'mean_dynamic_minus_baseline': delta.mean(),
                     'dynamic_better': int((delta < -1e-12).sum()),
                     'tied': int((delta.abs() <= 1e-12).sum()),
                     'dynamic_worse': int((delta > 1e-12).sum())})
    wins = pd.DataFrame(wins)
    weights = pd.DataFrame([json.loads(value) for value in dynamic.model_weights])
    distribution = expected.groupby(['split','model_name']).agg(
        mean_actual_error=('target_error','mean'), mean_predicted_error=('predicted_error','mean'),
        mean_outcome_scale=('outcome_scale','mean')).reset_index()
    historical = expected[expected.split.eq('train')].groupby('model_name').target_error.mean()
    chosen = selective[(selective.split=='test') & (selective.location=='ALL')].iloc[0]

    figure_dir = root/'artifacts/figures'/manifest['run_id']
    figures = []
    def save(fig, name):
        path = figure_dir/name
        save_figure(fig, path)
        figures.append(str(path.relative_to(root)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in zip(axes, ['normalized_mae', 'mae'], ['Normalized MAE', 'MAE (monthly cases)']):
        comparison[metric].rename(index=LABELS).plot.bar(ax=ax, color=['#3b82b6']*4+['#b74e36'])
        ax.set(title='Test: 21 pseudo interventions', ylabel=label, xlabel='', ylim=(0,None))
        ax.tick_params(axis='x', rotation=35)
    save(fig, 'cde_test_comparison.png')

    fig, ax = plt.subplots(figsize=(11,5))
    per_area.rename(columns=LABELS).plot.bar(ax=ax)
    ax.set(title='Test comparison by zone', ylabel='Mean normalized MAE', xlabel='Zone', ylim=(0,None))
    ax.tick_params(axis='x', rotation=0)
    ax.legend(fontsize=8)
    save(fig, 'cde_test_by_zone.png')

    fig, ax = plt.subplots(figsize=(8,5))
    for kind, group in tuning.groupby('model_kind'):
        ax.plot(group.temperature, group.validation_normalized_mae, 'o-', label=kind)
    ax.set(xscale='log', xlabel='Softmax temperature', ylabel='Validation mean normalized MAE',
           title='Model and temperature selected using validation only')
    ax.legend(fontsize=8)
    save(fig, 'cde_validation_selection.png')

    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, (kind, column) in zip(axes, [('Gradient boosting','gradient_boosting_error_prediction'),
                ('Histogram gradient boosting','hist_gradient_boosting_error_prediction'),
                ('Historical average','historical_error_prediction')]):
        for name, group in test_expected.groupby('model_name'):
            ax.scatter(group.target_error, group[column], label=LABELS[name], alpha=.7)
        extent = max(test_expected.target_error.max(), test_expected[column].max())*1.05
        ax.plot([0,extent],[0,extent],'--',color='grey')
        ax.set(title=kind, xlabel='Actual normalized MAE', ylabel='Predicted normalized MAE',
               xlim=(0,extent), ylim=(0,extent))
        ax.legend(fontsize=8)
    fig.suptitle('Test expected-error predictions (42 estimator records)')
    save(fig, 'cde_expected_error.png')

    fig, ax = plt.subplots(figsize=(12,5))
    weights.rename(columns=LABELS).plot.bar(stacked=True, ax=ax)
    labels = [f'{r.treated_location} {r.pseudo_date[:7]} h{r.horizon}' for r in dynamic.itertuples()]
    ax.set_xticklabels(labels, fontsize=7, rotation=90)
    ax.set(title='Frozen reliability model: test weights', ylabel='Weight', xlabel='Pseudo intervention', ylim=(0,1))
    save(fig, 'cde_dynamic_weights.png')

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(risk.coverage,risk.retained_error,'o-',label='Test curve (descriptive only)')
    if pd.notna(chosen.retained_error):
        ax.scatter([chosen.coverage],[chosen.retained_error],marker='*',s=200,color='#b74e36',
                   zorder=5,label='Validation-selected operating point')
    ax.set(title='Test error versus coverage; no test threshold tuning', xlabel='Retained fraction',
           ylabel='Mean retained normalized MAE', xlim=(0,1.03), ylim=(0,None))
    ax.legend(fontsize=8)
    save(fig, 'cde_error_coverage.png')

    fig, ax = plt.subplots(figsize=(11,5))
    deltas = paired[['dynamic_minus_cits','dynamic_minus_equal_weight']].copy()
    deltas.columns = ['Dynamic minus CITS', 'Dynamic minus equal weights']
    deltas.plot.bar(ax=ax)
    ax.axhline(0,color='black',linewidth=.8)
    ax.set(title='Paired test errors: positive means the proposed ensemble is worse',
           xlabel='Pseudo intervention (saved CSV order)', ylabel='Difference in normalized MAE')
    save(fig, 'cde_paired_errors.png')

    baseline_file = root/'artifacts/latest_run.json'
    if baseline_file.exists():
        baseline = json.loads(baseline_file.read_text())
        pd.read_csv(root/baseline['metrics_csv']).to_csv(reports/'baseline_results.csv',index=False)

    numerical = comparison.normalized_mae
    conclusion = ('The proposed ensemble does not beat all required baselines on the held-out mean normalized MAE. '
                  'The improvement hypothesis is not established.' if
                  any(numerical['dynamic'] >= numerical[name] for name in STRATEGIES[:-1]) else
                  'The proposed ensemble has the lowest mean normalized MAE in this exploratory test; '
                  'dependence and prior inspection prevent a confirmatory superiority claim.')
    error_lines = ['# Phase C/D/E error analysis', '', conclusion, '', markdown(wins), '',
        'Differences are paired by experiment; positive differences mean the proposed ensemble is worse. '
        'Win counts are descriptive, not independent Bernoulli trials.', '',
        f"CITS weights range from {weights.cits.min():.6f} to {weights.cits.max():.6f} "
        f"(mean {weights.cits.mean():.6f}). The selected temperature {manifest['temperature']} produces nearly "
        'equal weights, explaining the small difference from equal averaging. SCM was the lowest-error '
        'estimator in training; this ranking reverses in the test averages. These observations show limited '
        'transfer of the training ranking to the test period, without identifying a causal explanation.', '',
        markdown(distribution), '',
        'The selected error learner overpredicts test error on average. Different epidemic periods, historical '
        'normalization scales and area composition can affect transfer; these are possible explanations, '
        'not established causes. The table reports the actual shifts.', '',
        'Largest excess errors relative to CITS:', '',
        markdown(paired.nlargest(5,'dynamic_minus_cits')[['experiment_id','cits','dynamic',
            'dynamic_minus_cits','model_disagreement','outcome_scale']]), '',
        f"The frozen score threshold retains {int(chosen.retained_count)}/{int(chosen.analyses)} test experiments. "
        'No test error reduction from abstention is demonstrated at this operating point. '
        'The full error-coverage curve is descriptive and is not used to retune the threshold. '
        'No models or settings were changed in response to these errors.']
    (reports/'cde_error_analysis.md').write_text('\n'.join(error_lines)+'\n')

    lines = ['# PP1 summary — computed Phase C/D/E results', '',
        f"Run: `{manifest['run_id']}`; pseudo source: `{manifest['pseudo_run']}`.", '', conclusion, '',
        f"Data audit: {audit['rows']} monthly rows and {audit['locations']} Niteroi zones. "
        'Original source and audit are documented in `dataset_audit.md`. Existing CITS/SCM implementations '
        'and their baseline outputs were inspected and preserved.', '',
        '## Experiments and separation', '',
        f"Valid pseudo-intervention experiments: **{manifest['pseudo_experiments_completed']}**; "
        f"estimator records: **{manifest['model_rows']}**; fitting failures: **{pseudo['failures']}**.", '',
        markdown(splits.reset_index()), '',
        'Each experiment is one area/start/horizon, with paired CITS and SCM records. Minimum history is '
        '36 months; starts are quarterly; horizons are 3 or 6 months. The entire outcome window must precede '
        'the real intervention clean cutoff (Zone 1: December 2014; others: December 2016). '
        'Windows crossing split boundaries are purged. Related areas, horizons and estimators share calendar '
        'partitions. Training labels alone fit preprocessing and error models. Validation selects model, '
        'temperature and threshold. Calibration is reserved for existing interval calibration. '
        'Counterfactual baselines use expanding histories available at each pseudo date; meta-model parameters '
        'stay frozen. This is rolling retrospective evaluation, not a single-origin future forecast.', '',
        '## Features and expected error', '',
        'Numeric features: '+', '.join(f'`{c}`' for c in manifest['numeric_features'])+'.', '',
        'Categorical feature: `model_name` (one-hot encoded). Median imputation and missing indicators are '
        'fitted on training only. Omitted because entirely unavailable: '+', '.join(manifest['omitted_features'])+'. '
        'Constant among measured training values: '+', '.join(manifest['constant_numeric_features'])+'. '
        'Donor diagnostics are missing for CITS and constant for the one-donor SCM. Outcome scale is used '
        'to define the target and convert predictions to cases; it is not a learned input.', '',
        'Target: pseudo-post MAE / max(pre-period mean cases, 1). Features exclude target outcomes, '
        'errors, experiment IDs and split labels. Contemporaneous untreated-control observations enter '
        'retrospective prediction/disagreement/extrapolation. Individual expected errors are exported '
        'in normalized units and cases. The weighted predicted estimator error is a selection proxy; '
        'it is not a separately calibrated expectation of ensemble error.', '',
        '## Models and validation decisions', '',
        f"Selected family: **{manifest['selected_model']}**; softmax temperature: **{manifest['temperature']}**.", '',
        'Both families use squared-error loss, learning rate 0.05, depth 2, minimum leaf size 4 and seed 42. '
        'Gradient boosting uses 60 trees and subsample 1. Histogram gradient boosting uses 60 iterations, '
        '4 maximum leaves, L2 regularization 1, 255 bins and disabled internal early stopping. '
        'Full actual estimator parameters:', '', '```json',
        json.dumps(manifest['candidate_hyperparameters'],indent=2), '```', '',
        markdown(tuning), '',
        'Non-ML expected-error baseline: training mean normalized error for each estimator. '
        f"Training averages: {historical.to_dict()}. Fixed historically-best estimator: **{design['best_historical']}**.", '',
        'Weights are softmax(-predicted normalized error / temperature), recalculated for each experiment. '
        'The reliability score is 1 / (1 + weighted predicted error / median training target error). '
        'Model family and temperature minimize mean validation ensemble normalized MAE. The score threshold '
        'minimizes validation retained error subject to at least 50% validation coverage; ties prefer greater coverage. '
        'The threshold is frozen before test evaluation and applied using scores alone.', '',
        f"Threshold: **{manifest['selector']['threshold']:.12f}**; validation retained "
        f"**{manifest['selector']['validation_coverage']:.6%}**; retained validation error "
        f"**{manifest['selector']['validation_risk']:.6f}**.", '',
        '## Actual test comparisons', '', markdown(comparison.reset_index()), '',
        'Lower is better. MAE and RMSE are in monthly notified cases. Each entry is the mean of '
        'per-experiment metrics (RMSE is not pooled). Normalized MAE divides each experiment by '
        'its own pre-period mean count, floored at 1. All five methods use the same 21 windows.', '',
        'By area (normalized MAE):', '', markdown(per_area.reset_index()), '',
        'By horizon (normalized MAE):', '', markdown(per_horizon.reset_index()), '',
        'Both candidate ensembles, each using its own validation-selected temperature:', '',
        markdown(candidate_comparison.reset_index()), '',
        'Histogram results are reported for comparison; they do not change the validation-selected family.', '',
        '## Expected-error prediction and selective evaluation', '', markdown(error_quality), '',
        'Expected-error prediction metrics pool the 42 test estimator records, in normalized-error units. '
        'These differ from counterfactual case-prediction accuracy. Per-area rank correlations and high-error '
        'AUC are in `meta_metrics.csv`; AUC is unavailable when the test subset lacks both classes.', '',
        markdown(selective[(selective.split.isin(['validation','test'])) & selective.location.eq('ALL')]), '',
        'Score-based retained coverage is separate from the full evidence policy: diagnostic support checks, '
        'single-donor warnings and sparse calibration still apply. Retaining a pseudo experiment does not '
        'authorize a supported causal estimate. See `cde_error_analysis.md` for paired errors and transfer analysis.', '',
        '## Verification', '', '```json', json.dumps(tests,indent=2), '```', '',
        'Tests cover feature/target separation, pre-intervention eligibility, calendar separation, both learners, '
        'training-only preprocessing, validation-only threshold selection, frozen inference, artifact persistence '
        'and existing application behavior.', '',
        '## Methodological limitations', '',
        '- This 2015 test was already inspected in earlier work. This extension is exploratory; computational '
        'split separation does not make it a fresh confirmatory study.',
        '- There are only four test calendar origins, overlapping 3/6-month windows, shared controls and '
        'shared expanding histories. The 21 experiments are dependent; no independent-sample significance claim is made.',
        '- Only one permanent untreated donor is available. SCM has no donor-choice flexibility; donor-removal '
        'sensitivity and in-space checks are unavailable. Zone 1 has no test experiments.',
        '- Validation is one epidemic year. Selecting family, temperature and threshold on the same small '
        'validation set can overfit it; the coverage floor is a design setting, not a guarantee of test coverage or risk.',
        '- Missingness and several support diagnostics do not vary in these complete-data experiments. '
        'They cannot establish robustness or feature contributions.',
        '- Contemporaneous donor data make this retrospective counterfactual evaluation, not operational '
        'forecasting. Seasonal shifts, spillover and unmeasured confounding remain possible.',
        '- Pseudo-post prediction error is observable; real intervention counterfactual error and causal '
        'effect are not. Good pseudo prediction would not itself establish a causal claim.',
        f"- Existing empirical interval calibration contains {manifest['calibration_blocks']} calendar year; "
        'it has no exact conformal or causal coverage guarantee. This extension does not add uncertainty methods.',
        '- No external-site generalization, structural time-series extension, new robustness methods or '
        'post-test tuning is included in this run. Previous phase artifacts remain separate.', '',
        '## Generated artifacts', '',
        '- `reports/results_table.csv`: per-experiment comparisons, split labels and frozen selection decisions.',
        '- `reports/test_comparison.csv`: test mean case errors and normalized errors.',
        '- `reports/expected_error_comparison.csv`: error-prediction quality for both learners and the non-ML baseline.',
        '- `reports/cde_error_analysis.md`: actual paired failures and distribution comparisons.',
        f"- `{manifest['directory']}/`: frozen design, both models, non-ML baseline, diagnostics, predictions and evaluation tables.",
        *['- `'+name+'`' for name in figures], '',
        'Reproduce with `python -m src.pipeline.pseudo`, `python -m src.pipeline.train`, '
        '`python -m pytest -q --junitxml=artifacts/metrics/all_tests.xml`, then '
        '`python -m src.pipeline.research_reporting`. Work stops at Phase C/D/E.']
    (reports/'pp1_summary.md').write_text('\n'.join(lines)+'\n')
    from src.pipeline.dashboard_reporting import update as update_dashboard_status
    update_dashboard_status(root)
    output = {'run_id':manifest['run_id'], 'phase':'C/D/E', 'figures':figures, 'tests':tests,
              'conclusion':conclusion, 'test_comparison':comparison.reset_index().to_dict(orient='records')}
    write_json(directory/'report_manifest.json',output)
    return output


if __name__ == '__main__':
    print(json.dumps(generate(Path(__file__).resolve().parents[2]),indent=2))
