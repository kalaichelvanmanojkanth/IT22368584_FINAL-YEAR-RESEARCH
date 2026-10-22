import numpy as np
import pandas as pd
from src.config import ProjectConfig, ExperimentConfig
from src.evaluation.pseudo_interventions import calendar_split, run_pseudo_experiments, predict_window


def configuration():
    p = ProjectConfig(treated_units=['treated'], control_units=['donor_a','donor_b'], clean_pre_end={'treated':'2005-12-01'})
    e = ExperimentConfig(train_end='2003-12-01',validation_end='2004-06-01',calibration_end='2004-12-01',test_end='2005-12-01',horizons=[3,6],spacing_months=6)
    return p,e


def test_chronological_pseudo_split(panel):
    p,e = configuration()
    result = run_pseudo_experiments(panel,p,e)
    assert result.failures.empty
    assert set(result.rows.split) == {'train','validation','calibration','test'}
    assert (pd.to_datetime(result.rows.train_end) < pd.to_datetime(result.rows.pseudo_date)).all()
    assert result.rows.groupby('experiment_id').split.nunique().max() == 1
    assert result.rows.groupby('pseudo_date').split.nunique().max() == 1
    assert np.isfinite(result.rows.target_error).all()


def test_boundary_crossing_purged():
    _,e = configuration()
    assert calendar_split(pd.Timestamp('2003-10-01'),pd.Timestamp('2004-03-01'),e) == 'purged'


def test_meta_features_unchanged_by_future_targets(panel):
    p,e = configuration()
    before = run_pseudo_experiments(panel,p,e)
    panel.loc[(panel.date >= '2005-07-01') & panel.location_id.eq('treated'),'dengue_cases'] = 999999
    after = run_pseudo_experiments(panel,p,e)
    cols = ['pre_rmse','trend_mismatch','model_disagreement','extrapolation_features']
    a=before.rows[before.rows.pseudo_date.eq('2005-07-01')]
    b=after.rows[after.rows.pseudo_date.eq('2005-07-01')]
    np.testing.assert_equal(a[cols].to_numpy(),b[cols].to_numpy())
    assert not np.array_equal(a.post_mae,b.post_mae)


def test_partition_validation_rejects_relabelling_and_missing_pairs(panel):
    import pytest
    from src.evaluation.pseudo_interventions import validate_experiment_partitions
    from src.data.schema import DataError
    p,e=configuration();result=run_pseudo_experiments(panel,p,e)
    validate_experiment_partitions(result.rows,e)
    tampered=result.rows.copy();tampered.loc[tampered.split.eq('test'),'split']='train'
    with pytest.raises(DataError,match='calendar'): validate_experiment_partitions(tampered,e)
    with pytest.raises(DataError,match='Incomplete'): validate_experiment_partitions(result.rows.iloc[1:],e)
    duplicated=pd.concat([result.rows,result.rows.iloc[[0]]])
    with pytest.raises(DataError,match='Duplicate'): validate_experiment_partitions(duplicated,e)
