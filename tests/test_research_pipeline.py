from pathlib import Path
import json
import pandas as pd
import pytest
from src.pipeline.predict import load_bundle,analyze
from src.models.calibration import BlockResidualCalibration
from src.evaluation.selective_prediction import EvidencePolicy

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.real_data
def test_frozen_model_has_no_test_experiments():
    if not (ROOT/'artifacts/latest_reliability.json').exists(): pytest.skip('Train actual data first')
    bundle,manifest=load_bundle(ROOT)
    table=pd.read_csv(ROOT/manifest['directory']/'expected_errors.csv')
    test_ids=set(table[table.split.eq('test')].experiment_id)
    assert not test_ids & set(bundle['model'].training_experiment_ids)
    assert pd.Timestamp(bundle['model'].trained_until)<pd.to_datetime(table[table.split.eq('validation')].pseudo_date).min()
    assert all(table.groupby('pseudo_date').split.nunique()==1)
    strategies=pd.read_csv(ROOT/manifest['directory']/'strategy_results.csv')
    assert {'cits','synthetic_control','equal_weight','best_historical','dynamic'}<=set(strategies.strategy)


@pytest.mark.real_data
def test_actual_analysis_cannot_claim_support():
    if not (ROOT/'artifacts/latest_reliability.json').exists(): pytest.skip('Train actual data first')
    result=analyze(ROOT,'niteroi_zone_v3','zone_3',horizon_months=6)
    assert result['decision']=='INSUFFICIENT_EVIDENCE'
    assert result['effect_absolute'] is None
    assert result['exploratory_contrast']['label']
    assert result['provenance']['train_end']<'2017-01-01'


def test_calibration_refuses_test_targets():
    with pytest.raises(ValueError,match='calibration'):
        BlockResidualCalibration.fit(pd.DataFrame({'split':['test']}))


def test_policy_refuses_test_threshold_tuning():
    with pytest.raises(ValueError,match='validation'):
        EvidencePolicy.fit(pd.DataFrame({'split':['test']}),.75,1.,8)


@pytest.mark.real_data
def test_cde_frozen_selection_and_candidate_training():
    import numpy as np
    from src.models.reliability_meta import ReliabilityMetaLearner
    from src.evaluation.selective_prediction import ValidationSelector
    from src.pipeline.train import select_candidate
    pointer=ROOT/'artifacts/latest_reliability.json'
    if not pointer.exists(): pytest.skip('Train actual data first')
    manifest=json.loads(pointer.read_text())
    if manifest.get('phase')!='C/D/E': pytest.skip('Requires Phase C/D/E artifacts')
    directory=ROOT/manifest['directory']
    design=json.loads((directory/'frozen_design.json').read_text())
    rows=pd.read_csv(directory/'expected_errors.csv')
    summaries=pd.read_csv(directory/'strategy_results.csv')
    assert summaries.loc[summaries.strategy.ne('dynamic'),['predicted_error','reliability_score']].isna().all().all()
    train_ids=set(rows[rows.split.eq('train')].experiment_id)
    for kind in ['gradient_boosting','hist_gradient_boosting']:
        model=ReliabilityMetaLearner.load(directory/f'{kind}.joblib')
        assert set(model.training_experiment_ids)==train_ids
        assert model.hyperparameters()==design['candidate_hyperparameters'][kind]
        # Inference cannot update preprocessing or depend on test labels.
        test=rows[rows.split.eq('test')].copy()
        actual=model.predict_expected_error(test)
        test['target_error']=1e9;test['post_mae']=1e9
        np.testing.assert_array_equal(actual,model.predict_expected_error(test))
        np.testing.assert_allclose(actual,rows[rows.split.eq('test')][kind+'_error_prediction'])
    chosen=select_candidate(design['tuning'])
    assert chosen['model_kind']==manifest['selected_model']
    assert chosen['temperature']==manifest['temperature']
    validation=summaries[(summaries.split=='validation') & (summaries.strategy=='dynamic')]
    selector=ValidationSelector.fit(validation,manifest['experiment_config']['minimum_selective_coverage'])
    assert selector.threshold==pytest.approx(design['selector']['threshold'])
    assert set(selector.validation_experiment_ids)==set(validation.experiment_id)
    assert not set(selector.validation_experiment_ids)&set(rows[rows.split=='test'].experiment_id)
    test=summaries[(summaries.split=='test') & (summaries.strategy=='dynamic')]
    np.testing.assert_array_equal(selector.accept(test.reliability_score),test.selective_retained.astype(bool))
    # Every final label still predates its unit's conservative intervention cutoff.
    for unit,group in rows.groupby('treated_location'):
        assert pd.to_datetime(group.evaluation_end).max()<=pd.Timestamp(manifest['config']['clean_pre_end'][unit])
