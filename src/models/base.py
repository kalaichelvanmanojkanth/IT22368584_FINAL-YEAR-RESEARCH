"""Leakage-resistant contracts shared by counterfactual estimators."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.data.schema import DataError


class ModelError(RuntimeError):
    """An estimator could not produce a scientifically usable result."""


@dataclass
class CounterfactualResult:
    model_name: str
    dates: pd.DatetimeIndex
    counterfactual_mean: np.ndarray
    pre_fit_metrics: dict
    diagnostics: dict
    assumptions: list[str]
    warnings: list[str]
    lower_interval: np.ndarray | None = None
    upper_interval: np.ndarray | None = None
    outcome_unit: str = "monthly notified dengue cases"

    def __post_init__(self) -> None:
        self.counterfactual_mean = np.asarray(self.counterfactual_mean, dtype=float)
        if self.counterfactual_mean.shape != (len(self.dates),) or not np.isfinite(self.counterfactual_mean).all():
            raise ModelError("Nonfinite or misaligned counterfactual predictions")
        if (self.counterfactual_mean < 0).any():
            raise ModelError("Negative case counterfactual")
        if (self.lower_interval is None) != (self.upper_interval is None):
            raise ModelError("Both interval bounds must be supplied together")
        if self.lower_interval is not None:
            lower, upper = np.asarray(self.lower_interval), np.asarray(self.upper_interval)
            if lower.shape != self.counterfactual_mean.shape or upper.shape != lower.shape:
                raise ModelError("Interval dimensions do not match predictions")
            if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
                raise ModelError("Invalid interval bounds")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"date": self.dates, "model_name": self.model_name,
                             "counterfactual_mean": self.counterfactual_mean,
                             "lower_interval": self.lower_interval, "upper_interval": self.upper_interval})


class CounterfactualModel(ABC):
    """Fit pre data only; post treated outcomes are forbidden in prediction inputs."""
    model_name: str

    @abstractmethod
    def fit(self, pre_intervention_dataframe: pd.DataFrame, treated_unit: str,
            control_units: list[str], config: ModelConfig, *, intervention_date: pd.Timestamp,
            clean_pre_end: pd.Timestamp | None = None) -> "CounterfactualModel": ...

    @abstractmethod
    def predict_counterfactual(self, post_dates: pd.DatetimeIndex, *, control_data: pd.DataFrame,
                               treated_population: pd.Series | None = None) -> CounterfactualResult: ...

    def diagnostics(self) -> dict[str, Any]:
        self._require_fitted()
        return dict(self._diagnostics)

    def summary(self) -> dict:
        self._require_fitted()
        return {"model_name": self.model_name, "treated_unit": self.treated_unit,
                "control_units": self.control_units, "training_start": str(self.pre_dates.min().date()),
                "training_end": str(self.pre_dates.max().date()), "intervention_date": str(self.cutoff.date()),
                "config": self.config.model_dump(), "diagnostics": self.diagnostics(),
                "assumptions": self.assumptions, "warnings": self.warnings}

    def _require_fitted(self) -> None:
        if not getattr(self, "fitted_", False):
            raise ModelError("Model has not been successfully fitted")

    def _prepare_fit(self, df: pd.DataFrame, treated_unit: str, control_units: list[str],
                     config: ModelConfig, intervention_date: pd.Timestamp,
                     clean_pre_end: pd.Timestamp | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.fitted_ = False
        self.config, self.treated_unit, self.control_units = config, treated_unit, list(control_units)
        self.cutoff = pd.Timestamp(intervention_date)
        self.warnings: list[str] = []
        if pd.isna(self.cutoff) or not self.cutoff.is_month_start:
            raise DataError("Intervention date must be a valid month start")
        if not control_units or treated_unit in control_units or len(set(control_units)) != len(control_units):
            raise DataError("No valid donors: require nonempty, unique controls disjoint from treated unit")
        required = {"study_id", "location_id", "date", "dengue_cases", "population", "control_status", "intervention_status", "intervention_start"}
        if missing := required - set(df):
            raise DataError(f"Model input missing columns: {sorted(missing)}")
        data = df.copy()
        data["date"] = pd.to_datetime(data["date"], errors="raise")
        if data.empty or data["date"].isna().any() or (data["date"] >= self.cutoff).any():
            raise DataError("Post-treatment leakage: fit accepts nonempty pre-intervention rows only")
        if clean_pre_end is not None and (data["date"] > pd.Timestamp(clean_pre_end)).any():
            raise DataError("Rows exceed the explicitly declared clean pre-period")
        if data["study_id"].nunique() != 1:
            raise DataError("Cross-site pooling is not supported by this baseline")
        self.study_id = str(data["study_id"].iloc[0])
        if not set(data["location_id"]) == {treated_unit, *control_units}:
            raise DataError("Fit data must contain exactly the declared treated unit and donors")
        if data.duplicated(["location_id", "date"]).any():
            raise DataError("Duplicate unit/month model input")
        if not data["intervention_status"].eq(0).all():
            raise DataError("Post-treatment leakage: exposed rows cannot enter fit")
        source_start = pd.to_datetime(data["intervention_start"], errors="coerce")
        if (source_start.notna() & data["date"].ge(source_start)).any():
            raise DataError("Post-treatment leakage: source intervention dates precede fit observations")
        controls = data[data["location_id"].isin(control_units)]
        if not controls["control_status"].eq(True).all() or controls["intervention_start"].notna().any():
            raise DataError("Donors must be permanent source-defined untreated controls")
        treated = data[data["location_id"].eq(treated_unit)].sort_values("date").set_index("date")
        if treated["control_status"].eq(True).any():
            raise DataError("Treated unit incorrectly labelled as a control")
        self.pre_dates = pd.DatetimeIndex(treated.index)
        expected = pd.date_range(self.pre_dates.min(), self.pre_dates.max(), freq="MS")
        if not self.pre_dates.equals(expected) or len(expected) < config.min_pre_months:
            raise DataError("Insufficient or noncontiguous pre-period history")
        for _, group in data.groupby("location_id"):
            if not pd.DatetimeIndex(group["date"].sort_values()).equals(expected):
                raise DataError("Treated and control pre-period calendars must match exactly")
        for column in ["dengue_cases", "population"]:
            values = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any() or (column == "population" and (values <= 0).any()):
                raise DataError(f"Missing/nonfinite/invalid {column}; no implicit imputation")
        if (data["dengue_cases"].to_numpy(float) % 1 != 0).any():
            raise DataError("Count models require integer observations")
        self.last_population = float(treated["population"].iloc[-1])
        self.training_observed = treated["dengue_cases"].to_numpy(float)
        self.training_population = treated["population"].to_numpy(float)
        self._diagnostics: dict = {}
        return treated, controls

    def _prepare_prediction(self, post_dates: pd.DatetimeIndex, control_data: pd.DataFrame,
                            treated_population: pd.Series | None) -> tuple[pd.DatetimeIndex, pd.DataFrame, np.ndarray]:
        self._require_fitted()
        dates = pd.DatetimeIndex(post_dates)
        if dates.empty or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing or (dates < self.cutoff).any():
            raise DataError("Prediction dates must be unique, ordered and at/after intervention")
        if not dates.equals(pd.date_range(dates.min(), periods=len(dates), freq="MS")):
            raise DataError("Prediction horizon must comprise contiguous month starts")
        controls = control_data.copy()
        required = {"study_id", "location_id", "date", "dengue_cases", "population", "control_status", "intervention_status"}
        if not required <= set(controls):
            raise DataError("Incomplete post-control schema")
        if self.treated_unit in set(controls["location_id"]):
            raise DataError("Post-treatment leakage: never pass treated outcomes to predict")
        if set(controls["location_id"]) != set(self.control_units) or set(controls["study_id"]) != {self.study_id}:
            raise DataError("Prediction donors/study differ from fitted donors/study")
        if not controls["control_status"].eq(True).all() or not controls["intervention_status"].eq(0).all():
            raise DataError("Contaminated or invalid post-period controls")
        if "intervention_start" in controls and controls["intervention_start"].notna().any():
            raise DataError("Post-period donor has a source intervention date")
        controls["date"] = pd.to_datetime(controls["date"], errors="raise")
        if controls.duplicated(["location_id", "date"]).any():
            raise DataError("Duplicate post-control observations")
        for _, group in controls.groupby("location_id"):
            if not pd.DatetimeIndex(group["date"].sort_values()).equals(dates):
                raise DataError("Post-control observations do not exactly cover prediction dates")
        for column in ["dengue_cases", "population"]:
            values = pd.to_numeric(controls[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any() or (column == "population" and (values <= 0).any()):
                raise DataError(f"Missing/invalid post-control {column}")
        if treated_population is None:
            population = np.full(len(dates), self.last_population)
            self.warnings.append("Post population not supplied; last pre-period population held constant")
        else:
            if not isinstance(treated_population, pd.Series) or not pd.DatetimeIndex(treated_population.index).equals(dates):
                raise DataError("Population covariate must be a Series indexed by exact prediction dates")
            population = treated_population.to_numpy(float)
        if not np.isfinite(population).all() or (population <= 0).any():
            raise DataError("Invalid treated population covariate")
        return dates, controls, population

    @staticmethod
    def donor_matrix(data: pd.DataFrame, dates: pd.DatetimeIndex, controls: list[str], outcome: str) -> np.ndarray:
        data = data.copy()
        if outcome == "dengue_incidence":
            data[outcome] = 100_000 * data["dengue_cases"] / data["population"]
        return data.pivot(index="date", columns="location_id", values=outcome).loc[dates, controls].to_numpy(float)
