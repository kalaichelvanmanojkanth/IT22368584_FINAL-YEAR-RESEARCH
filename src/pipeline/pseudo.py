"""Persist the real-data historical experiment table and all failures."""
import argparse
import logging
from pathlib import Path
from src.config import load_config,load_experiments
from src.data.preprocess import preprocess
from src.evaluation.pseudo_interventions import run_pseudo_experiments
from src.utils.reproducibility import run_manifest,write_json


def run(root: Path):
    project=load_config(root/'configs/niteroi.yaml')
    experiment=load_experiments(root/'configs/experiments.yaml')
    panel,_=preprocess(root,project)
    manifest=run_manifest(root,project.model_dump(mode='json'),root/project.data.raw_dir/project.data.filename)
    manifest.update(phase='C',experiment_config=experiment.model_dump())
    directory=root/'artifacts/runs'/manifest['run_id']; directory.mkdir(parents=True)
    result=run_pseudo_experiments(panel,project,experiment)
    for name,table in [('experiments',result.rows),('trajectories',result.predictions),('failures',result.failures)]:
        table.to_csv(directory/f'{name}.csv',index=False)
    manifest.update(experiments=int(result.rows.experiment_id.nunique()) if not result.rows.empty else 0,
                    model_rows=len(result.rows),failures=len(result.failures),directory=str(directory.relative_to(root)))
    write_json(directory/'manifest.json',manifest)
    write_json(root/'artifacts/latest_pseudo.json',manifest)
    logging.info('Pseudo experiments: %s', {k:manifest[k] for k in ['experiments','model_rows','failures','directory']})
    if result.rows.empty: raise RuntimeError('No valid pseudo experiments; inspect failures.csv')
    return manifest

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    run(Path(__file__).resolve().parents[2])
