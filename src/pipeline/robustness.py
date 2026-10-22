"""Run frozen-model robustness evaluations; never use results for model selection."""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import load_niteroi
from src.evaluation.placebo import in_time_placebos,in_space_placebos,leave_one_donor_out
from src.evaluation.stress_tests import stress_scenarios,run_stress,semi_synthetic
from src.models.reliability_meta import ReliabilityMetaLearner,FEATURES
from src.pipeline.predict import load_bundle
from src.pipeline.train import aggregate
from src.utils.reproducibility import write_json


def run(root: Path):
    project=load_config(root/'configs/niteroi.yaml'); panel=load_niteroi(root,project.data)
    bundle,manifest=load_bundle(root); directory=root/manifest['directory']
    pseudo=json.loads((root/'artifacts/latest_pseudo.json').read_text())
    if pseudo['run_id']!=bundle['pseudo_run']: raise ValueError('Pseudo artifact differs from frozen reliability run')
    original=pd.read_csv(root/pseudo['directory']/'trajectories.csv')
    in_time_placebos(original).to_csv(directory/'placebos.csv',index=False)
    dates=pd.date_range('2015-01-01',periods=6,freq='MS')
    in_space_placebos(panel,dates,project).to_csv(directory/'in_space_placebos.csv',index=False)
    selected=panel[panel.location_id.isin(['zone_2',*project.control_units])]
    pre=selected[selected.date<dates[0]];post=selected[selected.date.isin(dates)]
    leave_one_donor_out(pre,post[post.location_id.isin(project.control_units)],
        post[post.location_id.eq('zone_2')].set_index('date').loc[dates,'population'],'zone_2',dates,project).to_csv(directory/'donor_sensitivity.csv',index=False)
    eligible=[unit for unit in project.treated_units if pd.Timestamp(project.clean_pre_end[unit])>=dates[-1]]
    stress=pd.DataFrame([run_stress(panel,project,bundle,unit,dates[0],scenario,project.seed)
                        for unit in eligible for scenario in stress_scenarios()])
    stress.to_csv(directory/'stress_tests.csv',index=False)
    series=pd.read_csv(directory/'strategy_trajectories.csv')
    synthetic=[]
    for identifier,group in series[(series.split=='test') & (series.strategy=='dynamic')].groupby('experiment_id'):
        for reduction in [0.,.25,.5]:
            result=semi_synthetic(group.observed,group.prediction,float(group.outcome_scale.iloc[0]),bundle['calibration'],reduction)
            synthetic.append(result|{'experiment_id':identifier,'location':group.treated_location.iloc[0]})
    pd.DataFrame(synthetic).to_csv(directory/'semi_synthetic.csv',index=False)
    rows=pd.read_csv(directory/'expected_errors.csv'); ablations=[]
    # Fixed model settings and previously selected temperature: diagnostic robustness only.
    for name,removed in {'without_missingness':['missing_rate'],
                         'without_donor_stability':['max_donor_weight','effective_donor_count','leave_one_donor_sensitivity'],
                         'without_disagreement':['model_disagreement'],
                         'without_placebo':['placebo_error']}.items():
        informative=[c for c in removed if rows[rows.split=='train'][c].nunique(dropna=True)>1]
        if not informative:
            ablations.append({'ablation':name,'status':'not_identifiable','reason':'Removed diagnostics are unavailable or constant in this dataset','test_normalized_mae':None})
            continue
        learner=ReliabilityMetaLearner(project.seed,[c for c in FEATURES if c not in removed]).fit(rows[rows.split=='train'])
        result,_=aggregate(rows[rows.split=='test'],original[original.split=='test'],learner,bundle['temperature'],bundle['best_historical'])
        ablations.append({'ablation':name,'status':'completed','reason':'Fixed full-model temperature; no test tuning',
                          'test_normalized_mae':float(result[result.strategy=='dynamic'].normalized_mae.mean())})
    pd.DataFrame(ablations).to_csv(directory/'ablations.csv',index=False)
    output={'reliability_run':manifest['run_id'],'in_time_rows':len(in_time_placebos(original)),
            'in_space_status':'unavailable with one permanent control','donor_removal_status':'unavailable with one permanent control',
            'stress_scenarios':len(stress),'stress_completed':int(stress.status.eq('completed').sum()),
            'stress_abstained_before_prediction':int(stress.status.eq('failed').sum()),'semi_synthetic_experiments':len(synthetic),
            'ablation_rows':len(ablations),'warning':'Exploratory robustness on frozen test experiments; no model reselection'}
    write_json(directory/'robustness.json',output); logging.info('%s',output)
    return output

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO);run(Path(__file__).resolve().parents[2])
