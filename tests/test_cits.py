import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.models.cits import ControlledInterruptedTimeSeries
from src.models.base import ModelError


def fit_cits(panel, family="negative_binomial"):
    cutoff = pd.Timestamp("2006-01-01")
    model = ControlledInterruptedTimeSeries().fit(panel[panel.date < cutoff], "treated", ["donor_a"],
             ModelConfig(cits_family=family), intervention_date=cutoff)
    return model


@pytest.mark.parametrize("family", ["negative_binomial", "poisson"])
def test_predictions_finite(panel, family):
    panel = panel[panel.location_id.ne("donor_b")]
    model = fit_cits(panel, family)
    dates = pd.date_range("2006-01-01", periods=12, freq="MS")
    controls = panel[(panel.date >= dates[0]) & panel.location_id.eq("donor_a")]
    result = model.predict_counterfactual(dates, control_data=controls,
                                         treated_population=pd.Series(100_000., index=dates))
    assert np.isfinite(result.counterfactual_mean).all()
    assert (result.counterfactual_mean >= 0).all()
    assert result.lower_interval is None
    assert len(model.diagnostics()["residuals"]) == 72
    assert model.summary()["training_end"] == "2005-12-01"


def test_predict_before_fit_rejected(panel):
    with pytest.raises(ModelError, match="successfully fitted"):
        ControlledInterruptedTimeSeries().predict_counterfactual(pd.date_range("2006-01-01", periods=2, freq="MS"), control_data=panel)
