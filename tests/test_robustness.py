import numpy as np
from src.models.calibration import BlockResidualCalibration
from src.evaluation.stress_tests import semi_synthetic
from src.evaluation.placebo import in_space_placebos
from src.config import ProjectConfig


def test_known_effect_recovery_with_perfect_counterfactual():
    observed=np.array([10.,20.,30.]); calibration=BlockResidualCalibration(1,1,.1,[1])
    result=semi_synthetic(observed,observed,1,calibration,.5)
    assert result['true_injected_effect']==-30
    assert result['effect_bias']==0
    assert result['interval_coverage']
    np.testing.assert_array_equal(observed,[10,20,30])


def test_no_in_space_placebo_without_independent_control(panel):
    result=in_space_placebos(panel,[],ProjectConfig())
    assert result.status.iloc[0]=='unavailable'
