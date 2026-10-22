import pandas as pd
import pytest

from src.config import ModelConfig
from src.data.schema import DataError
from src.models.base import ModelError
from src.models.cits import ControlledInterruptedTimeSeries
from src.models.structural_ts import StructuralTimeSeries
from src.models.synthetic_control import SyntheticControl


def test_failed_refit_cannot_reuse_previous_model(panel):
    model = SyntheticControl().fit(panel[panel.date < "2006-01-01"], "treated", ["donor_a", "donor_b"],
                                  ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))
    with pytest.raises(DataError):
        model.fit(panel, "treated", ["donor_a", "donor_b"], ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))
    with pytest.raises(ModelError, match="successfully fitted"):
        model.summary()


def test_rank_deficient_cits_stops(panel):
    panel = panel[(panel.date < "2006-01-01") & panel.location_id.ne("donor_b")].copy()
    panel.loc[panel.location_id.eq("donor_a"), "dengue_cases"] = 10
    with pytest.raises(ModelError, match="rank deficient"):
        ControlledInterruptedTimeSeries().fit(panel, "treated", ["donor_a"], ModelConfig(),
                                             intervention_date=pd.Timestamp("2006-01-01"))


def test_pending_model_never_fakes_output():
    with pytest.raises(NotImplementedError, match="pending"):
        StructuralTimeSeries().predict_counterfactual([], control_data=pd.DataFrame())
