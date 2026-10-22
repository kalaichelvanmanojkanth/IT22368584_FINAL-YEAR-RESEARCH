"""Standalone recorded-intervention analysis with fail-closed evidence status."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import load_niteroi
from src.data.schema import DataError
from src.evaluation.metrics import effect_summary
from src.evaluation.pseudo_interventions import predict_window
from src.explanation.reliability_explanation import explain
from src.models.ensemble import error_weights,combine
from src.models.base import ModelError
from src.utils.reproducibility import write_json


def load_bundle(root: Path):
    path=root/'artifacts/latest_reliability.json'
    if not path.exists(): raise DataError('Train reliability first: python -m src.pipeline.train')
    manifest=json.loads(path.read_text())
    bundle=joblib.load(root/manifest['directory']/'bundle.joblib')
    project=load_config(root/'configs/niteroi.yaml')
    checksum=hashlib.sha256((root/project.data.raw_dir/project.data.filename).read_bytes()).hexdigest()
    if checksum!=bundle['dataset_sha256']: raise DataError('Current dataset differs from trained model provenance')
    if project.model_dump(mode='json')!=bundle['project_config']: raise DataError('Project config changed since training; regenerate experiments and train')
    return bundle,manifest


def analyze(root: Path, study_id: str, location_id: str, intervention_date: str | None=None,
            horizon_months: int=6) -> dict:
    project=load_config(root/'configs/niteroi.yaml')
    response={'study_id':study_id,'location_id':location_id,'intervention_type':'wMel Wolbachia mosquito deployments',
              'effect_absolute':None,'effect_percent':None,'interval_lower':None,'interval_upper':None,
              'reliability_score':None,'decision':'INSUFFICIENT_EVIDENCE','model_weights':{},
              'diagnostics':{},'assumptions':[],'warnings':[],'trajectories':[],
              'exploratory_contrast':None,'explanation':{}}
    if study_id!=project.data.study_id or location_id not in project.treated_units:
        raise DataError('Unsupported study or treated location; external adapters are not activated')
    if horizon_months<1 or horizon_months>24: raise DataError('Horizon must be between 1 and 24 months')
    try:
        bundle,manifest=load_bundle(root)
        panel=load_niteroi(root,project.data)
        target=panel[panel.location_id.eq(location_id)]
        start=pd.Timestamp(target.monitoring_start.iloc[0])
        if intervention_date is not None and pd.Timestamp(intervention_date)!=start:
            raise DataError('Use the recorded monitoring date; arbitrary intervention dates are not supported by this endpoint')
        dates=pd.date_range(start,periods=horizon_months,freq='MS')
        cutoff=pd.Timestamp(project.clean_pre_end[location_id])
        selected=panel[panel.location_id.isin([location_id,*project.control_units])]
        pre=selected[selected.date<=cutoff]
        post=selected[selected.date.isin(dates)]
        population=post[post.location_id.eq(location_id)].set_index('date').population.reindex(dates)
        results,features,models=predict_window(pre,post[post.location_id.isin(project.control_units)],population,location_id,dates,project)
        frame=pd.DataFrame([v|{'model_name':k} for k,v in features.items()]).sort_values('model_name')
        errors=bundle['model'].predict_expected_error(frame)
        weights=error_weights(errors,bundle['temperature'])
        prediction=combine([results[k].counterfactual_mean for k in frame.model_name],weights)
        diagnostics={c:float(frame[c].mean()) for c in ['pre_rmse','model_disagreement','missing_rate','extrapolation_features','gap_months','horizon']}
        diagnostics['effective_donor_count']=float(frame.effective_donor_count.dropna().min())
        diagnostics['predicted_error']=float(weights@errors)
        assessment=bundle['policy'].assess(diagnostics,bundle['calibration'].block_count)
        observed=target.set_index('date').dengue_cases.reindex(dates).to_numpy(float)
        scale=float(frame.outcome_scale.iloc[0])
        lower,upper=bundle['calibration'].interval(prediction,scale)
        contrast=effect_summary(observed,prediction)
        interval_low,interval_high=bundle['calibration'].effect_interval(observed,prediction,scale)
        response.update({k:assessment[k] for k in ['decision','reliability_score']})
        response.update(model_weights=dict(zip(frame.model_name,weights.tolist())),
                        diagnostics=diagnostics,explanation=explain(diagnostics,assessment,bundle),
                        provenance={'reliability_run':manifest['run_id'],'dataset_sha256':manifest['dataset_sha256'],
                                    'train_end':str(pre.date.max().date()),'monitoring_start':str(start.date()),
                                    'zone_wide_release_start':str(target.intervention_start.iloc[0].date())})
        response['diagnostics']['predicted_errors_by_model']=dict(zip(frame.model_name,errors.tolist()))
        response['diagnostics']['models']={k:m.summary() for k,m in models.items()}
        response['warnings']=list(dict.fromkeys(assessment['reasons']+[w for r in results.values() for w in r.warnings]))
        response['assumptions']=list(dict.fromkeys(a for r in results.values() for a in r.assumptions))
        response['warnings'].append('International Wolbachia estimates do not establish effects in Sri Lanka or effects of fogging/source reduction')
        response['exploratory_contrast']=contrast|{'interval_lower':interval_low,'interval_upper':interval_high,
            'label':'Unvalidated exploratory model contrast; do not use as a supported intervention conclusion'}
        if response['decision']=='SUPPORTED_ESTIMATE':
            response.update({k:contrast[k] for k in ['effect_absolute','effect_percent']})
            response.update(interval_lower=interval_low,interval_upper=interval_high)
        for i,date in enumerate(dates):
            response['trajectories'].append({'date':str(date.date()),'observed':float(observed[i]),
                'counterfactual':float(prediction[i]),'lower':float(lower[i]),'upper':float(upper[i]),
                **{k:float(v.counterfactual_mean[i]) for k,v in results.items()}})
    except (DataError,ModelError,ValueError,KeyError) as exc:
        response['warnings'].append(f'Analysis abstained: {exc}')
        response['explanation']={'reasons':response['warnings']}
    return response

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--location',default='zone_2'); parser.add_argument('--horizon',type=int,default=6)
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    project=load_config(root/'configs/niteroi.yaml')
    result=analyze(root,project.data.study_id,args.location,horizon_months=args.horizon)
    path=root/'artifacts/predictions'/f'reliability_{args.location}_{args.horizon}.json'
    write_json(path,result)
    print(json.dumps({'decision':result['decision'],'warnings':result['warnings'],'output':str(path)},indent=2))
