import pandas as pd
from src.evaluation.selective_prediction import EvidencePolicy,risk_coverage
from src.models.calibration import BlockResidualCalibration


def test_abstention_on_unreliable_input():
    p=EvidencePolicy(1,1,1,1,1,0,3,6,8)
    assert p.assess({'predicted_error':.2,'gap_months':20,'horizon':6},20)['decision']=='INSUFFICIENT_EVIDENCE'
    assert p.assess({'predicted_error':.2,'gap_months':0,'horizon':6},1)['decision']=='UNCERTAIN_REVIEW_REQUIRED'
    assert p.assess({'predicted_error':.2,'gap_months':0,'horizon':6},20)['decision']=='SUPPORTED_ESTIMATE'


def test_calendar_blocks_not_rows():
    frame=pd.DataFrame({'date':['2001-01-01','2001-02-01','2001-01-01'],'split':'calibration',
                        'observed':[10,12,13],'prediction':[9,9,9],'outcome_scale':1})
    c=BlockResidualCalibration.fit(frame)
    assert c.block_count==1
    assert c.quantile==4
    low,high=c.effect_interval([10,12],[9,9],1)
    assert low==-4 and high==12


def test_risk_coverage():
    result=risk_coverage([1,5],[.9,.1])
    assert result.iloc[-1].coverage==.5
    assert result.iloc[-1].retained_error==1


def test_validation_selector_freezes_threshold():
    import numpy as np
    import pytest
    from src.evaluation.selective_prediction import ValidationSelector
    validation=pd.DataFrame({'experiment_id':['a','b','c','d'],'split':'validation',
                             'normalized_mae':[1.,2.,10.,20.],'reliability_score':[.9,.8,.2,.1]})
    selector=ValidationSelector.fit(validation,.5)
    assert selector.threshold==.8
    assert selector.validation_coverage==.5
    assert selector.validation_risk==1.5
    assert selector.evaluate(validation)['retained_count']==2
    with pytest.raises(ValueError,match='validation'):
        ValidationSelector.fit(validation.assign(split='test'),.5)
    altered=validation.assign(normalized_mae=[100,100,0,0])
    np.testing.assert_array_equal(selector.accept(validation.reliability_score),selector.accept(altered.reliability_score))
    empty=selector.evaluate(validation.assign(reliability_score=.01))
    assert empty['coverage']==0
    assert empty['retained_error'] is None


def test_validation_threshold_applies_to_evidence_status():
    p=EvidencePolicy(1,1,1,1,1,0,3,6,8,reliability_threshold=.9)
    result=p.assess({'predicted_error':1.,'horizon':6},20)
    assert result['decision']=='INSUFFICIENT_EVIDENCE'
    assert any('threshold' in reason for reason in result['reasons'])
