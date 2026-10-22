"""Explicitly artificial stress tests and semi-synthetic known-effect recovery."""
import numpy as np
import pandas as pd

from src.data.schema import DataError
from src.evaluation.metrics import prediction_metrics,effect_summary
from src.evaluation.pseudo_interventions import predict_window
from src.models.base import ModelError
from src.models.ensemble import error_weights,combine


def stress_scenarios() -> list[dict]:
    return [{'name':'unmodified reference','kind':'reference','value':0}]+[
        {'name':f'missing_{int(rate*100)}percent','kind':'missing','value':rate} for rate in [.05,.1,.2]]+[
        {'name':f'history_{months}months','kind':'history','value':months} for months in [24,36,60]]+[
        {'name':'remove_strongest_donor','kind':'remove_donor','value':1},
        {'name':'control_reporting_inflation_25percent','kind':'contamination','value':1.25}]+[
        {'name':f'horizon_{months}months','kind':'horizon','value':months} for months in [3,6]]


def run_stress(panel,project,bundle,unit,start,scenario,seed=42) -> dict:
    config=project.model_copy(deep=True)
    horizon=int(scenario['value']) if scenario['kind']=='horizon' else 6
    dates=pd.date_range(start,periods=horizon,freq='MS')
    selected=panel[panel.location_id.isin([unit,*config.control_units])]
    pre=selected[selected.date<dates[0]].copy(); post=selected[selected.date.isin(dates)].copy()
    result={'analysis_type':'artificial stress test on real clean historical trajectories','location':unit,
            'pseudo_date':str(dates[0].date()),'scenario':scenario['name'],'kind':scenario['kind'],'value':scenario['value'],
            'status':'failed','decision':'INSUFFICIENT_EVIDENCE','mae':None,'normalized_mae':None,'error':None}
    try:
        if dates[-1]>pd.Timestamp(config.clean_pre_end[unit]): raise DataError('Stress target is outside clean history')
        if scenario['kind']=='missing':
            rng=np.random.default_rng(seed)
            indices=rng.choice(pre.index,size=max(1,round(len(pre)*scenario['value'])),replace=False)
            pre.loc[indices,'dengue_cases']=np.nan
        elif scenario['kind']=='history':
            pre=pre[pre.date>=dates[0]-pd.DateOffset(months=int(scenario['value']))]
        elif scenario['kind']=='remove_donor':
            # Compute strongest donor from pre-only SCM, without looking at future outcomes.
            from src.models.synthetic_control import SyntheticControl
            model=SyntheticControl().fit(pre,unit,config.control_units,config.models,intervention_date=dates[0])
            removed=max(model.diagnostics()['donor_weights'],key=model.diagnostics()['donor_weights'].get)
            config.control_units=[c for c in config.control_units if c!=removed]
            pre=pre[pre.location_id.isin([unit,*config.control_units])]
            post=post[post.location_id.isin([unit,*config.control_units])]
        elif scenario['kind']=='contamination':
            mask=post.location_id.isin(config.control_units)
            post.loc[mask,'dengue_cases']=np.round(post.loc[mask,'dengue_cases']*scenario['value'])
        controls=post[post.location_id.isin(config.control_units)]
        population=post[post.location_id.eq(unit)].set_index('date').loc[dates,'population']
        results,features,_=predict_window(pre,controls,population,unit,dates,config,missing_rate=float(pre.dengue_cases.isna().mean()))
        frame=pd.DataFrame([v|{'model_name':k} for k,v in features.items()]).sort_values('model_name')
        errors=bundle['model'].predict_expected_error(frame); weights=error_weights(errors,bundle['temperature'])
        prediction=combine([results[k].counterfactual_mean for k in frame.model_name],weights)
        observed=post[post.location_id.eq(unit)].set_index('date').loc[dates,'dengue_cases'].to_numpy(float)
        metrics=prediction_metrics(observed,prediction); scale=float(frame.outcome_scale.iloc[0])
        diagnostics={c:float(frame[c].mean()) for c in ['pre_rmse','model_disagreement','missing_rate','extrapolation_features','gap_months','horizon']}
        diagnostics.update(predicted_error=float(weights@errors),effective_donor_count=float(frame.effective_donor_count.dropna().min()))
        assessment=bundle['policy'].assess(diagnostics,bundle['calibration'].block_count)
        result.update(status='completed',mae=metrics['mae'],normalized_mae=metrics['mae']/scale,
                      decision=assessment['decision'],error=' | '.join(assessment['reasons']))
    except (DataError,ModelError,ValueError,KeyError) as exc:
        result['error']=str(exc)
    return result


def semi_synthetic(observed,prediction,scale,calibration,reduction: float) -> dict:
    """Fractional injected reductions are methodological trajectories, not case records."""
    if not 0<=reduction<=1: raise ValueError('Reduction must be between zero and one')
    original=np.asarray(observed,float); prediction=np.asarray(prediction,float)
    modified=original*(1-reduction)
    truth=float(np.sum(modified-original))
    estimate=effect_summary(modified,prediction)['effect_absolute']
    lower,upper=calibration.effect_interval(modified,prediction,scale)
    return {'analysis_type':'semi-synthetic experiment','injected_reduction':reduction,
            'true_injected_effect':truth,'estimated_effect':estimate,'effect_bias':estimate-truth,
            'absolute_effect_error':abs(estimate-truth),'relative_effect_error':abs(estimate-truth)/abs(truth) if abs(truth)>1e-8 else None,
            'interval_lower':lower,'interval_upper':upper,'interval_coverage':bool(lower<=truth<=upper),'interval_width':upper-lower}
