"""Pre-fitted controlled ITS counterfactual, not a post-outcome treatment regression."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import ModelConfig
from src.explanation.diagnostics import pre_diagnostics
from src.models.base import CounterfactualModel, CounterfactualResult, ModelError


class ControlledInterruptedTimeSeries(CounterfactualModel):
    """NB/Poisson GLM: population offset, trend, Fourier seasonality, donor rates.

    All coefficients use the clean pre-period only. Intervention indicator is set
    to zero for Y(0); no treatment coefficient is estimable from pre-only data.
    Actual post outcomes enter effect contrasts outside this estimator.
    """
    model_name = "cits"

    def _design(self, dates: pd.DatetimeIndex, controls: pd.DataFrame) -> pd.DataFrame:
        months = (dates.year - self.pre_dates[0].year) * 12 + dates.month - self.pre_dates[0].month
        columns = {"intercept": np.ones(len(dates)), "trend_years": np.asarray(months, float) / 12}
        for harmonic in range(1, self.config.seasonal_harmonics + 1):
            phase = 2 * np.pi * harmonic * np.asarray(months) / 12
            columns[f"sin_{harmonic}"] = np.sin(phase)
            columns[f"cos_{harmonic}"] = np.cos(phase)
        rates = self.donor_matrix(controls, dates, self.control_units, "dengue_incidence")
        for i, unit in enumerate(self.control_units):
            columns[f"log1p_rate_{unit}"] = np.log1p(rates[:, i])
        return pd.DataFrame(columns, index=dates)

    def fit(self, pre_intervention_dataframe: pd.DataFrame, treated_unit: str,
            control_units: list[str], config: ModelConfig, *, intervention_date: pd.Timestamp,
            clean_pre_end: pd.Timestamp | None = None) -> "ControlledInterruptedTimeSeries":
        treated, controls = self._prepare_fit(pre_intervention_dataframe, treated_unit, control_units,
                                             config, intervention_date, clean_pre_end)
        design = self._design(self.pre_dates, controls)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ModelError("CITS design matrix is rank deficient; revise controls/harmonics")
        family = (sm.families.NegativeBinomial(alpha=config.nb_alpha)
                  if config.cits_family == "negative_binomial" else sm.families.Poisson())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                self.regression_ = sm.GLM(treated["dengue_cases"].to_numpy(float), design,
                                          family=family, offset=np.log(self.training_population)).fit(maxiter=300)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ModelError(f"CITS fitting failed: {exc}") from exc
        self.warnings.extend(str(w.message) for w in caught)
        if not self.regression_.converged or not np.isfinite(self.regression_.params).all():
            raise ModelError("CITS failed to converge to finite parameters")
        self.fitted_pre_ = np.asarray(self.regression_.predict(design, offset=np.log(self.training_population)))
        if not np.isfinite(self.fitted_pre_).all():
            raise ModelError("CITS produced nonfinite fitted values")
        self._diagnostics = pre_diagnostics(self.training_observed, self.fitted_pre_)
        self._diagnostics.update({"coefficients": self.regression_.params.to_dict(),
                                  "converged": bool(self.regression_.converged),
                                  "design_condition_number": float(np.linalg.cond(design)),
                                  "family": config.cits_family,
                                  "nb_alpha": config.nb_alpha if config.cits_family == "negative_binomial" else None,
                                  "counterfactual_intervention_indicator": 0})
        self.assumptions = ["Pre-period treated/control relationship remains stable without intervention",
                            "Control outcomes are unaffected by intervention and spillover",
                            "Population denominators and surveillance definitions are comparable over time",
                            "Post control outcomes are permitted for retrospective evaluation, not prospective forecasting"]
        self.warnings += ["Pre-fitted CITS variant; not a replication of the publication's full segmented regression",
                          "Base estimator has no native calibrated interval; ensemble calibration must be assessed separately",
                          "Treatment effect coefficient is not fitted; effect is an observed-minus-Y(0) contrast"]
        if config.cits_family == "negative_binomial":
            self.warnings.append("Negative-binomial alpha is fixed by config, not estimated or tuned on the post period")
        self.fitted_ = True
        return self

    def predict_counterfactual(self, post_dates: pd.DatetimeIndex, *, control_data: pd.DataFrame,
                               treated_population: pd.Series | None = None) -> CounterfactualResult:
        dates, controls, population = self._prepare_prediction(post_dates, control_data, treated_population)
        predictions = np.asarray(self.regression_.predict(self._design(dates, controls), offset=np.log(population)))
        return CounterfactualResult(self.model_name, dates, predictions,
                                    self._diagnostics["pre_fit_metrics"], self.diagnostics(),
                                    list(self.assumptions), list(self.warnings))
