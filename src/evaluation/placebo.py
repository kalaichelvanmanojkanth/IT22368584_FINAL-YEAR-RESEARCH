"""In-time, in-space and donor deletion checks with design eligibility checks."""
import numpy as np
import pandas as pd
from src.config import ProjectConfig
from src.evaluation.pseudo_interventions import predict_window
from src.models.synthetic_control import SyntheticControl


def in_time_placebos(experiments: pd.DataFrame) -> pd.DataFrame:
    """Historical windows have no known intervention: report signed apparent effects."""
    frame=experiments.copy()
    frame['residual']=frame.observed-frame.prediction
    return frame.groupby(['experiment_id','treated_location','split','model_name']).agg(
        apparent_effect=('residual','sum'),mean_absolute_error=('residual',lambda x:float(np.abs(x).mean())),
        months=('date','count')).reset_index().assign(analysis_type='in-time placebo; no intervention applied')


def leave_one_donor_out(pre,controls,population,unit,dates,project: ProjectConfig) -> pd.DataFrame:
    rows=[]
    for removed in project.control_units:
        donors=[x for x in project.control_units if x!=removed]
        if not donors:
            rows.append({'removed_donor':removed,'status':'unavailable','reason':'No untreated donors remain','mean_absolute_prediction_change':None})
            continue
        full=SyntheticControl().fit(pre,unit,project.control_units,project.models,intervention_date=dates[0])
        full_prediction=full.predict_counterfactual(dates,control_data=controls,treated_population=population)
        reduced=SyntheticControl().fit(pre[pre.location_id.isin([unit,*donors])],unit,donors,project.models,intervention_date=dates[0])
        prediction=reduced.predict_counterfactual(dates,control_data=controls[controls.location_id.isin(donors)],treated_population=population)
        rows.append({'removed_donor':removed,'status':'completed','reason':None,
                     'mean_absolute_prediction_change':float(np.abs(prediction.counterfactual_mean-full_prediction.counterfactual_mean).mean())})
    return pd.DataFrame(rows)


def in_space_placebos(panel,dates,project: ProjectConfig) -> pd.DataFrame:
    """Only permanently untreated targets with another untreated donor qualify."""
    if len(project.control_units)<2:
        return pd.DataFrame([{'status':'unavailable','reason':'Only one permanently untreated zone; no independent donor remains'}])
    rows=[]
    for unit in project.control_units:
        donors=[x for x in project.control_units if x!=unit]
        copy=panel[panel.location_id.isin([unit,*donors])].copy()
        # Pseudo-target designation in the experiment copy only; original observations preserved.
        copy.loc[copy.location_id.eq(unit),'control_status']=False
        config=project.model_copy(deep=True); config.control_units=donors
        pre=copy[copy.date<dates[0]]; post=copy[copy.date.isin(dates)]
        results,_,_=predict_window(pre,post[post.location_id.isin(donors)],
                post[post.location_id.eq(unit)].set_index('date').loc[dates,'population'],unit,dates,config)
        actual=post[post.location_id.eq(unit)].set_index('date').loc[dates,'dengue_cases'].to_numpy(float)
        for name,result in results.items(): rows.append({'location':unit,'model_name':name,'status':'completed',
                         'apparent_effect':float(np.sum(actual-result.counterfactual_mean))})
    return pd.DataFrame(rows)
