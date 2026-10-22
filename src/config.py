"""Strict, portable configuration for the Phase A/B pipeline."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfig):
    study_id: str = "niteroi_zone_v3"
    country: str = "Brazil"
    raw_dir: Path = Path("data/raw/niteroi")
    filename: str = "Niteroi_ZoneData_May2021_Shared.xls"
    article_id: int = 13662203
    version: int = 3
    expected_md5: str = "a31b89e0f2bc0a4d17271b3a386defb8"


class ModelConfig(StrictConfig):
    min_pre_months: int = Field(36, ge=24)
    min_post_months: int = Field(6, ge=1)
    horizon_months: int = Field(6, ge=1)
    seasonal_harmonics: int = Field(2, ge=0, le=5)
    cits_family: Literal["negative_binomial", "poisson"] = "negative_binomial"
    nb_alpha: float = Field(1.0, gt=0)
    scm_outcome: Literal["dengue_incidence", "dengue_cases"] = "dengue_incidence"
    donor_concentration_warning: float = Field(0.8, gt=0, le=1)
    optimizer_max_iterations: int = Field(2000, ge=1)
    optimizer_tolerance: float = Field(1e-10, gt=0)
    holdout_months: int = Field(12, ge=1)


class ProjectConfig(StrictConfig):
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    seed: int = Field(42, ge=0)
    treated_units: list[str] = ["zone_1", "zone_2", "zone_3", "zone_4"]
    control_units: list[str] = ["control"]
    # Conservative pre-period ends before any pilot release in the treated zone.
    clean_pre_end: dict[str, str] = {
        "zone_1": "2014-12-01", "zone_2": "2016-12-01",
        "zone_3": "2016-12-01", "zone_4": "2016-12-01",
    }

    @model_validator(mode="after")
    def disjoint_units(self) -> "ProjectConfig":
        if not self.control_units or not self.treated_units:
            raise ValueError("At least one treated unit and control unit are required")
        if set(self.treated_units) & set(self.control_units):
            raise ValueError("Treated and control units must be disjoint")
        if len(set(self.control_units)) != len(self.control_units):
            raise ValueError("Control IDs must be unique")
        return self


def load_config(path: str | Path) -> ProjectConfig:
    """Read YAML; all data paths are resolved against the repository by the CLI."""
    return ProjectConfig.model_validate(yaml.safe_load(Path(path).read_text()))


class ExperimentConfig(StrictConfig):
    """Calendar partitions shared by all areas, horizons and estimators."""
    minimum_pre_history: int = Field(36, ge=24)
    horizons: list[int] = [3, 6]
    spacing_months: int = Field(3, ge=1)
    missingness_tolerance: float = Field(0.0, ge=0, le=1)
    train_end: str = '2012-12-01'
    validation_end: str = '2013-12-01'
    calibration_end: str = '2014-12-01'
    test_end: str = '2015-12-01'
    temperatures: list[float] = [0.1, 0.3, 1.0, 3.0]
    acceptance_quantile: float = Field(0.75, gt=0, lt=1)
    minimum_selective_coverage: float = Field(0.5, gt=0, le=1)
    interval_alpha: float = Field(0.1, gt=0, lt=1)
    minimum_calibration_blocks: int = Field(8, ge=2)
    seed: int = 42

    @model_validator(mode='after')
    def valid_design(self):
        import pandas as pd
        boundaries = [pd.Timestamp(getattr(self, name)) for name in
                      ['train_end', 'validation_end', 'calibration_end', 'test_end']]
        if any(pd.isna(d) or not d.is_month_start for d in boundaries) or boundaries != sorted(set(boundaries)):
            raise ValueError('Split boundaries must be increasing unique month starts')
        if not self.horizons or any(h < 1 for h in self.horizons) or len(set(self.horizons)) != len(self.horizons):
            raise ValueError('Horizons must be positive and unique')
        if not self.temperatures or any(t <= 0 for t in self.temperatures):
            raise ValueError('Temperatures must be positive')
        return self


def load_experiments(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
