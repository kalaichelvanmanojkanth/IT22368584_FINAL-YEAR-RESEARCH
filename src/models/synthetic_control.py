"""Classic simplex-constrained donor matching with explicit optimisation failure."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import ModelConfig
from src.explanation.diagnostics import pre_diagnostics
from src.models.base import CounterfactualModel, CounterfactualResult, ModelError


class SyntheticControl(CounterfactualModel):
    model_name = "synthetic_control"

    def fit(self, pre_intervention_dataframe: pd.DataFrame, treated_unit: str,
            control_units: list[str], config: ModelConfig, *, intervention_date: pd.Timestamp,
            clean_pre_end: pd.Timestamp | None = None) -> "SyntheticControl":
        _, controls = self._prepare_fit(pre_intervention_dataframe, treated_unit, control_units,
                                        config, intervention_date, clean_pre_end)
        matrix = self.donor_matrix(controls, self.pre_dates, self.control_units, config.scm_outcome)
        target = self.training_observed.copy()
        if config.scm_outcome == "dengue_incidence":
            target = target / self.training_population * 100_000
        # Common positive scaling improves numerics without changing the minimizer.
        scale = max(float(np.std(target)), 1.)
        x, y = matrix / scale, target / scale
        donor_count = len(control_units)
        if donor_count == 1:
            weights = np.ones(1)  # the unique feasible solution, not failed optimisation
            message = "Single-donor simplex: analytic unique solution w=1"
            self.warnings.append("Only one donor: no learned donor diversity or leave-one-donor-out estimate")
        else:
            solution = minimize(lambda w: float(np.mean((y - x @ w) ** 2)),
                                np.full(donor_count, 1 / donor_count),
                                jac=lambda w: -2 * x.T @ (y - x @ w) / len(y),
                                method="SLSQP", bounds=[(0., 1.)] * donor_count,
                                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1,
                                              "jac": lambda w: np.ones_like(w)}],
                                options={"maxiter": config.optimizer_max_iterations,
                                         "ftol": config.optimizer_tolerance})
            if not solution.success:
                raise ModelError(f"Synthetic Control optimisation failed: {solution.message}")
            weights, message = solution.x, str(solution.message)
        if not np.isfinite(weights).all() or np.min(weights) < -1e-8 or abs(weights.sum() - 1) > 1e-7:
            raise ModelError("Synthetic Control returned invalid simplex weights")
        self.weights_ = np.maximum(weights, 0)
        self.weights_ /= self.weights_.sum()
        fitted = matrix @ self.weights_
        self.fitted_pre_ = fitted * self.training_population / 100_000 if config.scm_outcome == "dengue_incidence" else fitted
        self._diagnostics = pre_diagnostics(self.training_observed, self.fitted_pre_)
        self._diagnostics.update({"donor_weights": dict(zip(self.control_units, self.weights_.tolist())),
                                  "max_donor_weight": float(self.weights_.max()),
                                  "donor_count": donor_count,
                                  "effective_donor_count": float(1 / np.sum(self.weights_ ** 2)),
                                  "optimisation_converged": True, "optimisation_message": message,
                                  "matching_outcome": config.scm_outcome,
                                  "matching_rmse": float(np.sqrt(np.mean((target - fitted) ** 2))),
                                  "leave_one_donor_sensitivity": None})
        if self.weights_.max() >= config.donor_concentration_warning:
            self.warnings.append("Severe donor concentration: largest weight exceeds configured diagnostic threshold")
        self.warnings.append("Base estimator has no native calibrated interval")
        self.assumptions = ["The untreated trajectory lies sufficiently close to the donor convex hull",
                            "Donors are unaffected by intervention/spillover and share surveillance changes",
                            "Pre-period donor weights remain applicable after intervention",
                            "Incidence matching is converted to case counts using supplied population" if config.scm_outcome == "dengue_incidence"
                            else "Raw case matching assumes comparable population scales"]
        self.fitted_ = True
        return self

    def predict_counterfactual(self, post_dates: pd.DatetimeIndex, *, control_data: pd.DataFrame,
                               treated_population: pd.Series | None = None) -> CounterfactualResult:
        dates, controls, population = self._prepare_prediction(post_dates, control_data, treated_population)
        result = self.donor_matrix(controls, dates, self.control_units, self.config.scm_outcome) @ self.weights_
        if self.config.scm_outcome == "dengue_incidence":
            result = result * population / 100_000
        return CounterfactualResult(self.model_name, dates, result,
                                    self._diagnostics["pre_fit_metrics"], self.diagnostics(),
                                    list(self.assumptions), list(self.warnings))
