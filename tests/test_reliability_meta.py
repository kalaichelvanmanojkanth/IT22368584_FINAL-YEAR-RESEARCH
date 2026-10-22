import numpy as np
import pandas as pd
import pytest
from src.models.reliability_meta import ReliabilityMetaLearner,FEATURES


def synthetic_experiments():
    # Explicit synthetic UNIT TEST diagnostics, not research results.
    rows=[]
    for i in range(30):
        rows.append({**{f:float(i%5) for f in FEATURES},'model_name':['cits','synthetic_control'][i%2],
                     'split':'train','target_error':float((i%5)+1),'experiment_id':f'UNIT_TEST_{i}',
                     'evaluation_end':'2001-12-01'})
    return pd.DataFrame(rows)


def test_meta_target_not_in_features():
    assert not {'post_mae','post_rmse','target_error','observed'} & set(FEATURES)
    with pytest.raises(ValueError,match='allowlist'): ReliabilityMetaLearner(features=['post_mae'])


def test_fit_refuses_test_data():
    frame=synthetic_experiments(); frame.loc[0,'split']='test'
    with pytest.raises(ValueError,match='training'): ReliabilityMetaLearner().fit(frame)


def test_reliability_bounds_and_roundtrip(tmp_path):
    frame=synthetic_experiments(); model=ReliabilityMetaLearner().fit(frame)
    score=model.predict_reliability(frame)
    assert np.all((score>=0)&(score<=1))
    model.save(tmp_path/'model.joblib')
    np.testing.assert_array_equal(model.predict_expected_error(frame),ReliabilityMetaLearner.load(tmp_path/'model.joblib').predict_expected_error(frame))
    altered=frame.copy(); altered['target_error']=999999
    np.testing.assert_array_equal(model.predict_expected_error(frame),model.predict_expected_error(altered))


@pytest.mark.parametrize('kind',['gradient_boosting','hist_gradient_boosting'])
def test_both_reliability_families_no_target_leakage(kind,tmp_path):
    frame=synthetic_experiments()
    model=ReliabilityMetaLearner(model_kind=kind).fit(frame)
    before=model.predict_expected_error(frame)
    altered=frame.copy();altered['target_error']=9999;altered['post_mae']=9999
    np.testing.assert_array_equal(before,model.predict_expected_error(altered))
    assert np.isfinite(before).all()
    if kind=='hist_gradient_boosting': assert model.hyperparameters()['early_stopping'] is False
    model.save(tmp_path/'candidate.joblib')
    np.testing.assert_array_equal(before,ReliabilityMetaLearner.load(tmp_path/'candidate.joblib').predict_expected_error(frame))


def test_non_ml_baseline_is_training_average():
    from src.models.reliability_meta import HistoricalErrorBaseline
    frame=synthetic_experiments()
    model=HistoricalErrorBaseline().fit(frame)
    expected=frame.groupby('model_name').target_error.mean()
    np.testing.assert_allclose(model.predict_expected_error(frame),frame.model_name.map(expected))
    frame['split']='test'
    with pytest.raises(ValueError,match='training'): HistoricalErrorBaseline().fit(frame)
