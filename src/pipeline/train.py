"""Train reliability on historical train only; tune weights on validation only."""
from dataclasses import asdict
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.config import load_config,load_experiments
from src.evaluation.metrics import prediction_metrics
from src.evaluation.selective_prediction import EvidencePolicy,risk_coverage,ValidationSelector
from src.models.calibration import BlockResidualCalibration
from src.models.ensemble import error_weights,combine
from src.models.reliability_meta import ReliabilityMetaLearner,FEATURES,HistoricalErrorBaseline
from src.utils.reproducibility import run_manifest,write_json
from src.evaluation.pseudo_interventions import validate_experiment_partitions


def aggregate(rows: pd.DataFrame, trajectories: pd.DataFrame, model: ReliabilityMetaLearner,
              temperature: float, best: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    summaries,series=[],[]
    for identifier,group in rows.groupby('experiment_id',sort=True):
        group=group.sort_values('model_name')
        source=trajectories[trajectories.experiment_id.eq(identifier)]
        matrix=source.pivot(index='date',columns='model_name',values='prediction').sort_index()[group.model_name].to_numpy().T
        observed=source.drop_duplicates('date').sort_values('date').observed.to_numpy(float)
        errors=model.predict_expected_error(group)
        dynamic=error_weights(errors,temperature)
        strategies={name:matrix[i] for i,name in enumerate(group.model_name)}
        strategies.update(equal_weight=matrix.mean(axis=0),best_historical=matrix[list(group.model_name).index(best)],
                          dynamic=combine(matrix,dynamic))
        scale=float(group.outcome_scale.iloc[0]); first=group.iloc[0]
        identity={key:first[key] for key in ['experiment_id','study_id','treated_location','pseudo_date','evaluation_end','horizon','split']}
        for name,prediction in strategies.items():
            metrics=prediction_metrics(observed,prediction)
            # This proxy belongs to the dynamic ensemble; do not attach its score
            # to baseline strategies as though it were their own error prediction.
            predicted_error=float(dynamic@errors) if name=='dynamic' else None
            summaries.append(identity|{'strategy':name,'mae':metrics['mae'],'rmse':metrics['rmse'],
                              'normalized_mae':metrics['mae']/scale,'predicted_error':predicted_error,
                              'reliability_score':1/(1+predicted_error/model.reference_error) if predicted_error is not None else None,
                              'outcome_scale':scale,
                              'model_weights':json.dumps(dict(zip(group.model_name,dynamic.tolist()))) if name=='dynamic' else None})
            series.append(pd.DataFrame({'date':sorted(source.date.unique()),'observed':observed,'prediction':prediction,
                                        'strategy':name,'outcome_scale':scale,**identity}))
    return pd.DataFrame(summaries),pd.concat(series,ignore_index=True)


def select_candidate(validation_results: list[dict]) -> dict:
    """Deterministic selection from validation metrics only; no test table accepted."""
    if not validation_results or any(not np.isfinite(r['validation_normalized_mae']) for r in validation_results):
        raise ValueError('Finite validation results required')
    return min(validation_results,key=lambda r:(r['validation_normalized_mae'],r['model_kind'],r['temperature']))


def train(root: Path) -> dict:
    project=load_config(root/'configs/niteroi.yaml'); cfg=load_experiments(root/'configs/experiments.yaml')
    pseudo=json.loads((root/'artifacts/latest_pseudo.json').read_text()); source=root/pseudo['directory']
    if pseudo['config'] != project.model_dump(mode='json') or pseudo['experiment_config'] != cfg.model_dump():
        raise ValueError('Configuration changed since pseudo generation; regenerate the source experiments')
    import hashlib
    if pseudo['dataset_sha256'] != hashlib.sha256((root/project.data.raw_dir/project.data.filename).read_bytes()).hexdigest():
        raise ValueError('Dataset changed since pseudo generation')
    rows=pd.read_csv(source/'experiments.csv'); trajectories=pd.read_csv(source/'trajectories.csv')
    validate_experiment_partitions(rows, cfg)
    for split in ['train','validation','calibration','test']:
        if rows[rows.split.eq(split)].empty: raise ValueError(f'Missing {split} experiments; no fallback split permitted')
    # Assert all outcome windows are ordered at meta-training boundaries.
    for left,right in [('train','validation'),('validation','calibration'),('calibration','test')]:
        if pd.to_datetime(rows[rows.split.eq(left)].evaluation_end).max()>=pd.to_datetime(rows[rows.split.eq(right)].pseudo_date).min():
            raise ValueError('Leaking chronological partitions')
    train_rows=rows[rows.split.eq('train')]; validation=rows[rows.split.eq('validation')].copy()
    historical_model=HistoricalErrorBaseline().fit(train_rows)
    historical=historical_model.errors
    best=historical_model.best_model
    candidates={kind:ReliabilityMetaLearner(cfg.seed,model_kind=kind).fit(train_rows)
                for kind in ['gradient_boosting','hist_gradient_boosting']}
    tuning=[]
    validation_series=trajectories[trajectories.split.eq('validation')]
    for kind,candidate in candidates.items():
        for candidate_temperature in cfg.temperatures:
            candidate_summary,_=aggregate(validation,validation_series,candidate,candidate_temperature,best)
            tuning.append({'model_kind':kind,'temperature':candidate_temperature,
                'validation_normalized_mae':float(candidate_summary[candidate_summary.strategy.eq('dynamic')].normalized_mae.mean())})
    choice=select_candidate(tuning)
    chosen_kind=choice['model_kind']; temperature=choice['temperature']; model=candidates[chosen_kind]
    validation['predicted_error']=model.predict_expected_error(validation)
    validation_summary,_=aggregate(validation,validation_series,model,temperature,best)
    validation_dynamic=validation_summary[validation_summary.strategy.eq('dynamic')]
    selector=ValidationSelector.fit(validation_dynamic,cfg.minimum_selective_coverage)
    policy=EvidencePolicy.fit(validation,cfg.acceptance_quantile,model.reference_error,cfg.minimum_calibration_blocks)
    policy.reliability_threshold=selector.threshold
    # Fit interval calibration without inspecting final test predictions or outcomes.
    cal_summary,cal_series=aggregate(rows[rows.split.eq('calibration')],trajectories[trajectories.split.eq('calibration')],model,temperature,best)
    calibration=BlockResidualCalibration.fit(cal_series[cal_series.strategy.eq('dynamic')],cfg.interval_alpha)
    manifest=run_manifest(root,project.model_dump(mode='json'),root/project.data.raw_dir/project.data.filename)
    directory=root/'artifacts/runs'/manifest['run_id']; directory.mkdir(parents=True)
    model_path=directory/'reliability.joblib'; model.save(model_path)
    for kind,candidate in candidates.items(): candidate.save(directory/f'{kind}.joblib')
    joblib.dump(historical_model,directory/'historical_error_baseline.joblib')
    bundle={'model':model,'temperature':temperature,'best_historical':best,'historical_errors':historical,
            'policy':policy,'calibration':calibration,'selector':selector,'model_kind':chosen_kind,'project_config':project.model_dump(mode='json'),
            'experiment_config':cfg.model_dump(),'pseudo_run':pseudo['run_id'],'dataset_sha256':manifest['dataset_sha256']}
    # Persist all decisions before test evaluation. No test-based refitting occurs below.
    joblib.dump(bundle,directory/'bundle.joblib')
    write_json(directory/'frozen_design.json',{'temperature':temperature,'best_historical':best,
              'policy':asdict(policy),'selector':asdict(selector),'calibration':asdict(calibration),'tuning':tuning,
              'training_until':model.trained_until,'selected_model':chosen_kind,
              'candidate_hyperparameters':{kind:candidate.hyperparameters() for kind,candidate in candidates.items()},
              'selection_objective':'Minimum mean validation ensemble normalized MAE; threshold minimizes validation retained risk subject to coverage floor',
              'test_status':'Previously inspected test set; exploratory extension, not fresh confirmatory evidence'})
    summary,series=aggregate(rows,trajectories,model,temperature,best)
    rows['predicted_error']=model.predict_expected_error(rows)
    rows['predicted_error_cases']=rows.predicted_error*rows.outcome_scale
    rows['historical_error_prediction']=historical_model.predict_expected_error(rows)
    for kind,candidate in candidates.items(): rows[kind+'_error_prediction']=candidate.predict_expected_error(rows)
    quality=[]
    high_error_threshold=float(validation.target_error.quantile(.75))
    for (split,unit),group in rows.groupby(['split','treated_location']):
        if split=='train': continue
        for predictor,column in [('meta_learner','predicted_error'),('historical_model_average','historical_error_prediction'),
                                 *[(kind,kind+'_error_prediction') for kind in candidates]]:
            metrics=prediction_metrics(group.target_error.to_numpy(),group[column].to_numpy())
            rank=spearmanr(group.target_error,group[column]).statistic if group[column].nunique()>1 and group.target_error.nunique()>1 else None
            labels=(group.target_error>high_error_threshold).astype(int)
            auc=float(roc_auc_score(labels,group[column])) if labels.nunique()==2 else None
            quality.append({'split':split,'location':unit,'predictor':predictor,**metrics,
                            'rank_correlation':float(rank) if rank is not None and np.isfinite(rank) else None,'high_error_auc':auc})
    intervals=[]
    for identifier,group in series[(series.split=='test') & (series.strategy=='dynamic')].groupby('experiment_id'):
        lower,upper=calibration.interval(group.prediction.to_numpy(),float(group.outcome_scale.iloc[0]))
        intervals.append({'experiment_id':identifier,'location':group.treated_location.iloc[0],
                          'empirical_trajectory_coverage':bool(np.all((group.observed>=lower)&(group.observed<=upper))),
                          'pointwise_coverage':float(np.mean((group.observed>=lower)&(group.observed<=upper))),
                          'mean_interval_width':float(np.mean(upper-lower))})
    for name,table in [('expected_errors',rows),('strategy_results',summary),('strategy_trajectories',series),
                        ('meta_metrics',pd.DataFrame(quality)),('temperature_validation',pd.DataFrame(tuning)),
                        ('test_interval_coverage',pd.DataFrame(intervals))]: table.to_csv(directory/f'{name}.csv',index=False)
    summary['selective_retained'] = None
    dynamic_mask=summary.strategy.eq('dynamic')
    summary.loc[dynamic_mask,'selective_retained']=selector.accept(summary.loc[dynamic_mask,'reliability_score'])
    summary['reliability_model']=chosen_kind
    summary.to_csv(directory/'strategy_results.csv',index=False)
    selective=[]
    for (split,unit),group in summary[dynamic_mask].groupby(['split','treated_location']):
        selective.append({'split':split,'location':unit,**selector.evaluate(group)})
    for split,group in summary[dynamic_mask].groupby('split'):
        selective.append({'split':split,'location':'ALL',**selector.evaluate(group)})
    pd.DataFrame(selective).to_csv(directory/'selective_evaluation.csv',index=False)
    risk_coverage(validation_dynamic.normalized_mae,validation_dynamic.reliability_score).to_csv(directory/'validation_risk_coverage.csv',index=False)
    candidate_test=[]
    for kind,candidate in candidates.items():
        candidate_choice=select_candidate([r for r in tuning if r['model_kind']==kind])
        candidate_result,_=aggregate(rows[rows.split=='test'],trajectories[trajectories.split=='test'],candidate,candidate_choice['temperature'],best)
        candidate_test.append(candidate_result[candidate_result.strategy=='dynamic'].assign(model_kind=kind,temperature=candidate_choice['temperature']))
    pd.concat(candidate_test,ignore_index=True).to_csv(directory/'candidate_test_results.csv',index=False)
    # Paired error analysis only: no changes to selection follow these test comparisons.
    test_table=summary[summary.split=='test'].pivot(index=['experiment_id','treated_location','pseudo_date','horizon'],columns='strategy',values='normalized_mae').reset_index()
    for strategy in ['cits','synthetic_control','equal_weight','best_historical']:
        test_table['dynamic_minus_'+strategy]=test_table.dynamic-test_table[strategy]
    diagnostics=rows[rows.split=='test'].groupby('experiment_id').agg(
        model_disagreement=('model_disagreement','first'),outcome_scale=('outcome_scale','first'))
    test_table=test_table.merge(diagnostics,on='experiment_id',validate='one_to_one')
    test_table.to_csv(directory/'paired_error_analysis.csv',index=False)
    selected=summary[(summary.split=='test') & (summary.strategy=='dynamic')]
    risk_coverage(selected.normalized_mae,selected.reliability_score).to_csv(directory/'risk_coverage.csv',index=False)
    manifest.update(phase='C/D/E',directory=str(directory.relative_to(root)),pseudo_run=pseudo['run_id'],
                    experiment_config=cfg.model_dump(),model_rows=len(rows),pseudo_experiments_completed=int(rows.experiment_id.nunique()),
                    reliability_model_trained=True,temperature=temperature,selected_model=chosen_kind,
                    selector=asdict(selector),calibration_blocks=calibration.block_count,
                    candidate_hyperparameters={kind:candidate.hyperparameters() for kind,candidate in candidates.items()},
                    numeric_features=model.available_features,categorical_features=['model_name'],
                    constant_numeric_features=[c for c in model.available_features if train_rows[c].nunique(dropna=True)<=1],
                    split_counts=rows.groupby('split').experiment_id.nunique().to_dict(),
                    supported_claim='Exploratory evaluation; hypothesis not established',
                    omitted_features=[c for c in FEATURES if c not in model.available_features])
    write_json(directory/'manifest.json',manifest); write_json(root/'artifacts/latest_reliability.json',manifest)
    logging.info('Reliability run %s; splits=%s; temperature=%s; calibration years=%s',manifest['run_id'],manifest['split_counts'],temperature,calibration.block_count)
    logging.info('Test strategies (normalized MAE): %s',selected[['treated_location','normalized_mae']].groupby('treated_location').mean().to_dict())
    return manifest

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO); train(Path(__file__).resolve().parents[2])
