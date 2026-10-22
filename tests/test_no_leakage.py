import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.data.schema import DataError
from src.models.cits import ControlledInterruptedTimeSeries
from src.models.synthetic_control import SyntheticControl


@pytest.mark.parametrize("model_type", [ControlledInterruptedTimeSeries, SyntheticControl])
def test_no_post_treatment_leakage(panel, model_type):
    panel = panel[panel.location_id.ne("donor_b")]
    with pytest.raises(DataError, match="leakage"):
        model_type().fit(panel, "treated", ["donor_a"], ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))


@pytest.mark.parametrize("model_type", [ControlledInterruptedTimeSeries, SyntheticControl])
def test_post_outcome_perturbation_cannot_change_prediction(panel, model_type):
    panel = panel[panel.location_id.ne("donor_b")].copy()
    cutoff = pd.Timestamp("2006-01-01")
    dates = pd.date_range(cutoff, periods=12, freq="MS")

    def predict(df):
        model = model_type().fit(df[df.date < cutoff], "treated", ["donor_a"], ModelConfig(), intervention_date=cutoff)
        controls = df[(df.date >= cutoff) & df.location_id.eq("donor_a")]
        return model.predict_counterfactual(dates, control_data=controls,
                        treated_population=pd.Series(100_000., index=dates)).counterfactual_mean

    before = predict(panel)
    panel.loc[(panel.date >= cutoff) & panel.location_id.eq("treated"), "dengue_cases"] = 999999
    np.testing.assert_array_equal(before, predict(panel))


@pytest.mark.parametrize("model_type", [ControlledInterruptedTimeSeries, SyntheticControl])
def test_prediction_rejects_treated_outcomes(panel, model_type):
    panel = panel[panel.location_id.ne("donor_b")]
    cutoff = pd.Timestamp("2006-01-01")
    model = model_type().fit(panel[panel.date < cutoff], "treated", ["donor_a"], ModelConfig(), intervention_date=cutoff)
    with pytest.raises(DataError, match="never pass treated outcomes"):
        model.predict_counterfactual(pd.date_range(cutoff, periods=12, freq="MS"), control_data=panel[panel.date >= cutoff])


def test_clean_pre_cutoff_enforced(panel):
    panel = panel[(panel.date < "2006-01-01") & panel.location_id.ne("donor_b")]
    with pytest.raises(DataError, match="clean pre-period"):
        SyntheticControl().fit(panel, "treated", ["donor_a"], ModelConfig(),
                               intervention_date=pd.Timestamp("2006-01-01"), clean_pre_end=pd.Timestamp("2004-12-01"))


def test_control_contamination_rejected(panel):
    panel = panel[(panel.date < "2006-01-01") & panel.location_id.ne("donor_b")]
    panel.loc[panel.location_id.eq("donor_a"), "control_status"] = False
    with pytest.raises(DataError, match="permanent"):
        SyntheticControl().fit(panel, "treated", ["donor_a"], ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))


def test_post_control_intervention_date_rejected(panel):
    panel = panel[panel.location_id.ne("donor_b")]
    cutoff = pd.Timestamp("2006-01-01")
    model = SyntheticControl().fit(panel[panel.date < cutoff], "treated", ["donor_a"], ModelConfig(), intervention_date=cutoff)
    controls = panel[(panel.date >= cutoff) & panel.location_id.eq("donor_a")].copy()
    controls["intervention_start"] = cutoff
    with pytest.raises(DataError, match="source intervention date"):
        model.predict_counterfactual(pd.date_range(cutoff, periods=12, freq="MS"), control_data=controls)
