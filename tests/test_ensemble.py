import numpy as np
import pytest
from src.models.ensemble import error_weights,combine


@pytest.mark.parametrize('strategy',['softmax','inverse'])
def test_ensemble_weights_sum_to_one(strategy):
    w=error_weights([0,0,1000],strategy=strategy)
    assert w.sum()==pytest.approx(1)
    assert (w>=0).all()
    assert np.isfinite(combine([[1,2],[2,3],[4,5]],w)).all()


def test_invalid_errors_rejected():
    with pytest.raises(ValueError): error_weights([np.nan,1])
    with pytest.raises(ValueError): error_weights([-1,1])
