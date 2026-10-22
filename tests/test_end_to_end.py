import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.config import load_config
from src.data.loaders import load_niteroi
from src.pipeline.baselines import analysis_window, run_baselines

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.real_data
def test_end_to_end_small_real_subset_if_data_available(tmp_path):
    config = load_config(ROOT / "configs/niteroi.yaml")
    source = ROOT / config.data.raw_dir / config.data.filename
    if not source.exists():
        pytest.skip("Original public source unavailable; no generated data substituted")
    destination = tmp_path / config.data.raw_dir
    destination.mkdir(parents=True)
    for path in source.parent.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            shutil.copy2(path, destination / path.name)
    for directory in ["artifacts/metrics", "artifacts/predictions", "reports"]:
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    config.treated_units = ["zone_2"]
    manifest = run_baselines(tmp_path, config, figures=False)
    assert manifest["completed_analyses"] == 4
    assert manifest["failed_analyses"] == 0
    table = pd.read_csv(tmp_path / manifest["metrics_csv"])
    assert table["reliability_score"].isna().all()
    assert table["interval_lower"].isna().all()
    assert table[table.analysis.eq("intervention")]["holdout_mae"].isna().all()
    assert table[table.analysis.eq("clean_pre_holdout")]["effect_absolute"].isna().all()
    assert (pd.to_datetime(table.train_end) < pd.to_datetime(table.evaluation_start)).all()
    assert (tmp_path / "reports/dataset_audit.md").exists()
    assert (tmp_path / "reports/pp1_summary.md").exists()
    prediction = pd.read_csv(tmp_path / manifest["predictions_csv"])
    assert prediction.counterfactual_mean.notna().all()
    assert json.loads((tmp_path / "artifacts/latest_run.json").read_text())["dataset_sha256"]


@pytest.mark.real_data
def test_real_holdout_is_within_clean_period():
    config = load_config(ROOT / "configs/niteroi.yaml")
    if not (ROOT / config.data.raw_dir / config.data.filename).exists():
        pytest.skip("Original source unavailable")
    panel = load_niteroi(ROOT, config.data)
    for unit in config.treated_units:
        pre, dates, cutoff, _ = analysis_window(panel, unit, config, "clean_pre_holdout")
        assert pre.date.max() < dates.min() == cutoff
        assert dates.max() <= pd.Timestamp(config.clean_pre_end[unit])
