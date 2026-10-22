"""Full research report and requested figures from actual computed artifacts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from src.pipeline.figures import plt,save_figure
from src.pipeline.reporting import test_summary
from src.pipeline.predict import analyze
from src.config import load_config
from src.utils.reproducibility import write_json

EXTENDED_METHODS='''

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
'''


def generate(root: Path) -> dict:
    manifest=json.loads((root/'artifacts/latest_reliability.json').read_text()); directory=root/manifest['directory']
    if manifest.get('phase') == 'C/D/E':
        from src.pipeline.cde_reporting import generate as generate_cde
        return generate_cde(root)
    summary=pd.read_csv(directory/'strategy_results.csv'); expected=pd.read_csv(directory/'expected_errors.csv')
    risk=pd.read_csv(directory/'risk_coverage.csv'); robust=json.loads((directory/'robustness.json').read_text())
    figure_dir=root/'artifacts/figures'/manifest['run_id']; figures=[]
    def save(fig,name):
        path=figure_dir/name;save_figure(fig,path);figures.append(str(path.relative_to(root)))
    test=expected[expected.split.eq('test')]
    fig,ax=plt.subplots(figsize=(8,6))
    for model,g in test.groupby('model_name'): ax.scatter(g.target_error,g.predicted_error,label=model,alpha=.7)
    extent=max(test.target_error.max(),test.predicted_error.max());ax.plot([0,extent],[0,extent],linestyle='--',color='grey',label='Perfect error prediction')
    ax.set(xlabel='Actual normalized MAE',ylabel='Predicted normalized MAE',title='Held-out expected-error predictions',xlim=(0,None),ylim=(0,None));ax.legend()
    save(fig,'06_expected_error_comparison.png')
    dynamic=summary[(summary.strategy=='dynamic') & (summary.split=='test')]
    weights=pd.DataFrame([json.loads(value) for value in dynamic.model_weights]);weights.index=dynamic.experiment_id
    fig,ax=plt.subplots(figsize=(12,5));weights.plot.bar(stacked=True,ax=ax)
    ax.set(xlabel='Historical test window',ylabel='Normalized model weight',title='Dynamic weights on held-out pseudo interventions',ylim=(0,1))
    ax.tick_params(axis='x',labelsize=6);ax.legend(title='Estimator');save(fig,'07_dynamic_model_weights.png')
    placebo=pd.read_csv(directory/'placebos.csv'); placebo=placebo[placebo.split.eq('test')]
    fig,ax=plt.subplots(figsize=(10,5))
    for model,g in placebo.groupby('model_name'): ax.scatter(g.experiment_id,g.apparent_effect,label=model,alpha=.7)
    ax.axhline(0,color='grey',linestyle='--',label='Null apparent effect');ax.set(xlabel='Historical test window',ylabel='Apparent total case difference',title='In-time placebo results (no intervention applied)')
    ax.tick_params(axis='x',rotation=90,labelsize=6);ax.legend();save(fig,'08_placebo_results.png')
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(risk.coverage,risk.retained_error,'o-',label='Dynamic ensemble')
    ax.set(xlabel='Retained proportion',ylabel='Mean normalized MAE',title='Held-out risk versus coverage',xlim=(0,1),ylim=(0,None));ax.legend();save(fig,'09_risk_coverage.png')
    stress=pd.read_csv(directory/'stress_tests.csv'); complete=stress[stress.status=='completed']
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    complete.pivot(index='scenario',columns='location',values='normalized_mae').plot.bar(ax=axes[0])
    axes[0].set(title='Computed artificial stress-test errors',xlabel='Artificial scenario',ylabel='Normalized MAE',ylim=(0,None));axes[0].tick_params(axis='x',rotation=60)
    counts=pd.crosstab(stress.scenario,stress.status);counts.plot.bar(stacked=True,ax=axes[1])
    axes[1].set(title='Completed and abstained scenarios',xlabel='Artificial scenario',ylabel='Number of analyses');axes[1].tick_params(axis='x',rotation=60)
    save(fig,'10_stress_tests.png')
    comparison=summary[summary.split=='test'].groupby(['treated_location','strategy']).normalized_mae.mean().unstack()
    fig,ax=plt.subplots(figsize=(11,5));comparison.plot.bar(ax=ax);ax.set(xlabel='Zone',ylabel='Mean normalized MAE',title='Final historical comparison: all required strategies',ylim=(0,None));ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=8)
    save(fig,'11_strategy_comparison.png')
    # Preserve A/B results as their own result table; current main table is the reliability evaluation.
    baseline=json.loads((root/'artifacts/latest_run.json').read_text())
    pd.read_csv(root/baseline['metrics_csv']).to_csv(root/'reports/baseline_results.csv',index=False)
    summary.to_csv(root/'reports/results_table.csv',index=False)
    configs=load_config(root/'configs/niteroi.yaml'); analyses=[]
    for unit in configs.treated_units:
        response=analyze(root,configs.data.study_id,unit,horizon_months=6)
        write_json(root/'artifacts/predictions'/f'reliability_{unit}_6.json',response)
        analyses.append({'location':unit,'decision':response['decision'],'reliability_score':response['reliability_score'],
                         'effect_absolute':response['effect_absolute'],'effect_percent':response['effect_percent'],
                         'reasons':' | '.join(response['explanation'].get('reasons',[]))})
    pd.DataFrame(analyses).to_csv(root/'reports/intervention_evidence.csv',index=False)
    test_counts=test_summary(root)
    audit=json.loads((root/'data/processed/validation.json').read_text())['statistics']
    lines=['# PP1 summary — computed research status','',f"Reliability run: {manifest['run_id']}",'',
        'Research question: Can reliability-aware counterfactual AI improve the dependability of intervention estimates while abstaining when evidence is insufficient?',
        '', f"Current data: original Niteroi Figshare zone-level v3 ({audit['rows']} monthly observations, {audit['locations']} zones). See dataset_audit.md for computed validation.",
        '',f"Completed pseudo windows: {manifest['pseudo_experiments_completed']}; estimator records: {manifest['model_rows']}.",
        f"Calendar partitions: {manifest['split_counts']}.",
        'Implemented: CITS and SCM, pseudo experiments, supervised gradient-boosting error model, dynamic ensemble, empirical calendar-block intervals, evidence policy, risk-coverage, placebos, stress tests, semi-synthetic recovery, Streamlit and optional FastAPI.',
        '', '## Held-out counterfactual comparison','',comparison.round(6).to_string(),'','Lower normalized MAE is better. Results are per area; windows and controls overlap.',
        '', 'The current experiment does NOT establish that dynamic weighting improves dependability. Compare every strategy above; no success claim is inferred from implementation.',
        '', '## Reliability prediction metrics','',pd.read_csv(directory/'meta_metrics.csv').query('split == "test"').round(6).to_string(index=False),
        '', 'High-error AUC is unavailable where validation-defined high-error classes are absent in the test subset. The score is not a probability of causality.',
        '', '## Robustness completed','',json.dumps(robust,indent=2),
        '', 'Semi-synthetic effect recovery:',pd.read_csv(directory/'semi_synthetic.csv').groupby('location')[['absolute_effect_error','interval_coverage','interval_width']].mean().to_string(),
        '', '## Current recorded-intervention evidence','',pd.DataFrame(analyses)[['location','decision','reliability_score']].to_string(index=False),
        '', 'Supported effect fields are withheld on abstention. Exploratory contrasts remain explicitly labelled in per-zone JSON artifacts.',
        '', '## Tests','',json.dumps(test_counts,indent=2),'',
        '## Artifacts','',f'- {manifest["directory"]}/ (models, frozen design, predictions, failures, diagnostics and evaluation tables)',
        '- artifacts/latest_pseudo.json (pseudo source run)', '- reports/intervention_evidence.csv','- reports/results_table.csv','- reports/baseline_results.csv']
    lines += ['- '+p for p in baseline['figure_paths']+figures]
    lines += ['','## Limitations and next research tasks','',
        f"Calibration contains {manifest['calibration_blocks']} distinct calendar year; no exact uncertainty coverage claim is justified.",
        'Only one untreated donor. In-space and donor-removal causal checks are not estimable here. Missingness stress currently causes safe abstention, not robust imputation.',
        'Unavailable/constant donor, placebo and missingness features do not establish meta-learning contributions. See ablations.csv.',
        'The structural time-series estimator remains explicitly pending. External source adapters and audits are implemented separately; intervention designs/outcome definitions prevent unreviewed pooling, and no external performance is claimed.',
        'Next research priorities: more independent untreated donors/calibration periods; source-reviewed external studies; compare revised prespecified designs on new evaluation data. Do not tune repeatedly on the reported final test.']
    (root/'reports/pp1_summary.md').write_text('\n'.join(lines)+'\n')
    method=root/'reports/methodology.md'
    original=method.read_text().split('## Phases C–H:')[0]
    original=original.replace('# Methodology — implemented Phases A/B','# Methodology — baseline specification and reliability extensions')
    original=original.replace('This hypothesis is untested. No reliability learner is trained yet.','A reliability learner is now trained; results are exploratory and do not establish the hypothesis.')
    # Baseline paragraphs describe native estimator behaviour. Explicitly mark their historical phase scope.
    marker='## Reproduction and deferred work'
    if marker in original: original=original.split(marker)[0]
    method.write_text(original+'\nThe preceding interval/status description refers to native Phase B baseline outputs. The current ensemble calibration and evidence policy are specified below.\n'+EXTENDED_METHODS)
    limitations=root/'reports/limitations.md'; text=limitations.read_text()
    text=text.replace('- No calibrated uncertainty, trained reliability model, selective-prediction results,\n  placebo validation or intervention-effect ground truth is currently available.',
        '- Reliability training, selective-prediction and empirical calibration are implemented; calibration has only one calendar year and real causal-effect ground truth remains unavailable.')
    text=text.replace('- Only baseline/audit figures are generated; later-phase figures are intentionally absent.',
        '- All requested figure categories are generated from saved computations; unavailable analyses are reported rather than plotted as successful results.')
    text += '\n- Dynamic weighting did not outperform the strongest baseline in the current held-out evaluation; the hypothesis is not established.\n' if 'Dynamic weighting did not outperform' not in text else ''
    text=text.replace('- Yogyakarta and Colombia adapters and design-comparability review are not implemented.', '- Yogyakarta and Colombia adapters preserve original sources separately. Different outcomes and unverified permanent controls preclude pooled evaluation; no external performance claim is made.')
    limitations.write_text(text)
    output={'run_id':manifest['run_id'],'figures':figures,'tests':test_counts,'supported_estimates':sum(a['decision']=='SUPPORTED_ESTIMATE' for a in analyses)}
    write_json(directory/'report_manifest.json',output)
    return output

if __name__=='__main__':
    print(json.dumps(generate(Path(__file__).resolve().parents[2]),indent=2))
