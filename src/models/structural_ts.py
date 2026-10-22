"""Structural time-series baseline: pending; no output is fabricated."""

IMPLEMENTATION_STATUS = "pending"

from src.models.base import CounterfactualModel


class StructuralTimeSeries(CounterfactualModel):
    """Reserved common interface; never returns an invented state-space estimate."""

    model_name = "structural_ts"

    def fit(self, pre_intervention_dataframe, treated_unit, control_units, config,
            *, intervention_date, clean_pre_end=None):
        raise NotImplementedError("Structural time-series baseline is pending; Phase B implements CITS and SCM")

    def predict_counterfactual(self, post_dates, *, control_data, treated_population=None):
        raise NotImplementedError("Structural time-series baseline is pending; no predictions available")
