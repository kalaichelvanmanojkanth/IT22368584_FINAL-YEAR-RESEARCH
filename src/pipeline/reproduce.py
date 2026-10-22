"""One-command reproduction from the pinned real source through report generation."""
import logging
from pathlib import Path
from src.config import load_config
from src.data.loaders import download_niteroi
from src.pipeline.baselines import run_baselines
from src.pipeline.pseudo import run as pseudo
from src.pipeline.train import train
from src.pipeline.robustness import run as robustness
from src.pipeline.research_reporting import generate


def main():
    logging.basicConfig(level=logging.INFO)
    root=Path(__file__).resolve().parents[2]; config=load_config(root/'configs/niteroi.yaml')
    download_niteroi(root,config.data)
    baseline=run_baselines(root,config)
    if baseline['failed_analyses']: raise RuntimeError('Baseline failures; inspect saved report before proceeding')
    pseudo(root);train(root);robustness(root);generate(root)
    logging.info('Completed standalone real-data reproduction. Hypothesis remains a question to evaluate.')

if __name__=='__main__': main()
