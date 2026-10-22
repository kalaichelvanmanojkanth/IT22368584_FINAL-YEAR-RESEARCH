"""Reproducible processing; never impute outcomes or rewrite raw data."""
import json
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.data.loaders import load_niteroi, study_metadata
from src.data.validate import ValidationReport, validate_panel, write_audit


def preprocess(root: Path, config: ProjectConfig) -> tuple[pd.DataFrame, ValidationReport]:
    panel = load_niteroi(root, config.data)
    report = validate_panel(panel, config.models.min_pre_months, config.models.min_post_months)
    metadata = study_metadata(config.data)
    write_audit(report, root / "reports/dataset_audit.md", metadata)
    report.raise_for_errors()
    directory = root / "data/processed"
    directory.mkdir(parents=True, exist_ok=True)
    panel.to_csv(directory / "niteroi.csv", index=False)
    (directory / "study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (directory / "validation.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return panel, report
