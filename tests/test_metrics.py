import numpy as np
import pytest
from src.evaluation.metrics import effect_summary, prediction_metrics


def test_relative_effect_zero_denominator():
    result = effect_summary(np.array([0., 1.]), np.zeros(2))
    assert result["effect_percent"] is None
    assert result["effect_absolute"] == 1
    assert result["estimated_cases_averted"] == -1


def test_effect_sign_and_missing_metrics():
    result = effect_summary(np.array([2., 2.]), np.array([4., 4.]))
    assert result["effect_absolute"] == -4
    assert result["effect_percent"] == -50
    with pytest.raises(ValueError, match="nonfinite"):
        prediction_metrics(np.array([np.nan]), np.array([1.]))


def test_mase_constant_scale_unavailable():
    assert prediction_metrics(np.array([1.]), np.array([2.]), np.ones(24))["mase"] is None
