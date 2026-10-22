from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import ModelConfig
from src.data.schema import DataError
from src.models.base import ModelError
from src.models.synthetic_control import SyntheticControl


def fit_scm(panel):
    return SyntheticControl().fit(panel[panel.date < "2006-01-01"], "treated", ["donor_a", "donor_b"],
                                  ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))


def test_scm_weights_nonnegative(panel):
    assert (fit_scm(panel).weights_ >= 0).all()


def test_scm_weights_sum_to_one(panel):
    model = fit_scm(panel)
    assert model.weights_.sum() == pytest.approx(1.)
    assert model.weights_[0] == pytest.approx(.5, abs=.02)
    assert model.diagnostics()["pre_fit_metrics"]["rmse"] < 1.


def test_scm_failed_optimisation_handled(panel, monkeypatch):
    monkeypatch.setattr("src.models.synthetic_control.minimize", lambda *args, **kwargs:
                        SimpleNamespace(success=False, message="deliberately forced unit-test failure"))
    with pytest.raises(ModelError, match="optimisation failed"):
        fit_scm(panel)


def test_no_donor_rejected(panel):
    with pytest.raises(DataError, match="No valid donors"):
        SyntheticControl().fit(panel, "treated", [], ModelConfig(), intervention_date=pd.Timestamp("2010-01-01"))


def test_single_donor_is_transparent(panel):
    pre = panel[(panel.date < "2006-01-01") & panel.location_id.ne("donor_b")]
    model = SyntheticControl().fit(pre, "treated", ["donor_a"], ModelConfig(), intervention_date=pd.Timestamp("2006-01-01"))
    assert model.weights_.tolist() == [1.]
    assert any("Only one donor" in warning for warning in model.warnings)


def test_missing_pre_outcomes_not_imputed(panel):
    panel.loc[0, "dengue_cases"] = np.nan
    with pytest.raises(DataError, match="imputation"):
        fit_scm(panel)
