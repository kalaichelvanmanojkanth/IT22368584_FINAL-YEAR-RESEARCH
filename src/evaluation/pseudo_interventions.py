"""Historical experiments: all targets follow fitting and feature extraction."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import ExperimentConfig, ProjectConfig
from src.data.schema import DataError
from src.evaluation.metrics import prediction_metrics
from src.explanation.diagnostics import reliability_features
from src.models.base import ModelError
from src.models.cits import ControlledInterruptedTimeSeries
from src.models.synthetic_control import SyntheticControl

MODELS = {'cits': ControlledInterruptedTimeSeries, 'synthetic_control': SyntheticControl}


@dataclass
class ExperimentResult:
    rows: pd.DataFrame
    predictions: pd.DataFrame
    failures: pd.DataFrame


def calendar_split(start: pd.Timestamp, end: pd.Timestamp, config: ExperimentConfig) -> str:
    """Purge windows crossing boundaries; never split related model/area rows randomly."""
    previous = pd.Timestamp.min
    for name, boundary in [('train', config.train_end), ('validation', config.validation_end),
                           ('calibration', config.calibration_end), ('test', config.test_end)]:
        boundary = pd.Timestamp(boundary)
        if start > previous and end <= boundary:
            return name
        previous = boundary
    return 'purged'


def generate_windows(panel: pd.DataFrame, project: ProjectConfig, experiments: ExperimentConfig):
    for unit in project.treated_units:
        source = panel[panel.location_id.eq(unit)].sort_values('date')
        if source.empty or unit not in project.clean_pre_end:
            raise DataError(f'Unknown unit or missing clean cutoff: {unit}')
        clean_end = min(pd.Timestamp(project.clean_pre_end[unit]), pd.Timestamp(experiments.test_end))
        if clean_end >= source.intervention_start.dropna().min():
            raise DataError('Clean cutoff must precede actual intervention')
        dates = pd.date_range(source.date.min(), clean_end, freq='MS')
        for index in range(experiments.minimum_pre_history, len(dates), experiments.spacing_months):
            for horizon in experiments.horizons:
                if index + horizon <= len(dates):
                    post = dates[index:index + horizon]
                    split = calendar_split(post[0], post[-1], experiments)
                    if split != 'purged':
                        yield unit, post, split


def predict_window(pre: pd.DataFrame, controls: pd.DataFrame, population: pd.Series,
                   unit: str, dates: pd.DatetimeIndex, project: ProjectConfig,
                   *, missing_rate: float = 0.0) -> tuple[dict, dict, dict]:
    """No post treated outcomes are accepted by this function."""
    results, features, models = {}, {}, {}
    for name, cls in MODELS.items():
        started = perf_counter()
        model = cls().fit(pre, unit, project.control_units, project.models,
                         intervention_date=dates[0], clean_pre_end=pre.date.max())
        result = model.predict_counterfactual(dates, control_data=controls, treated_population=population)
        results[name], models[name] = result, model
        features[name] = reliability_features(model, pre, controls, dates, missing_rate)
        features[name]['runtime_seconds'] = perf_counter() - started
    scale = features[next(iter(features))]['outcome_scale']
    disagreement = float(np.std([r.counterfactual_mean for r in results.values()], axis=0).mean() / scale)
    for row in features.values():
        row['model_disagreement'] = disagreement
    return results, features, models


def run_pseudo_experiments(panel: pd.DataFrame, project: ProjectConfig,
                           experiments: ExperimentConfig) -> ExperimentResult:
    rows, trajectories, failures = [], [], []
    project = project.model_copy(deep=True)
    project.models.min_pre_months = experiments.minimum_pre_history
    for unit, dates, split in generate_windows(panel, project, experiments):
        identity = {'experiment_id': f'{project.data.study_id}_{unit}_{dates[0]:%Y%m}_{len(dates)}',
                    'study_id': project.data.study_id, 'treated_location': unit,
                    'pseudo_date': str(dates[0].date()), 'evaluation_end': str(dates[-1].date()),
                    'horizon': len(dates), 'split': split}
        try:
            eligible = panel[panel.location_id.isin([unit, *project.control_units])]
            pre = eligible[eligible.date < dates[0]].copy()
            post = eligible[eligible.date.isin(dates)]
            missing_rate = float(pre.dengue_cases.isna().mean())
            if missing_rate > experiments.missingness_tolerance:
                raise DataError('Historical missingness exceeds configured tolerance')
            # Baselines deliberately reject any remaining missing value; tolerance is not imputation.
            control_data = post[post.location_id.isin(project.control_units)]
            population = post[post.location_id.eq(unit)].set_index('date').loc[dates, 'population']
            results, features, _ = predict_window(pre, control_data, population, unit, dates, project,
                                                  missing_rate=missing_rate)
            # Target outcomes are first read here, AFTER all features and predictions exist.
            actual = post[post.location_id.eq(unit)].set_index('date').loc[dates, 'dengue_cases'].to_numpy(float)
            for name, result in results.items():
                metrics = prediction_metrics(actual, result.counterfactual_mean)
                feature = features[name]
                rows.append(identity | feature | {'model_name': name, 'post_mae': metrics['mae'],
                            'post_rmse': metrics['rmse'], 'target_error': metrics['mae'] / feature['outcome_scale'],
                            'train_end': str(pre.date.max().date())})
                trajectories.append(pd.DataFrame({'date': dates, 'observed': actual,
                    'prediction': result.counterfactual_mean, 'model_name': name, **identity}))
        except (DataError, ModelError, KeyError, ValueError) as exc:
            failures.append(identity | {'error_type': type(exc).__name__, 'error': str(exc)})
    return ExperimentResult(pd.DataFrame(rows), pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(),
                            pd.DataFrame(failures, columns=['experiment_id','study_id','treated_location','pseudo_date',
                                'evaluation_end','horizon','split','error_type','error']))


def validate_experiment_partitions(rows: pd.DataFrame, config: ExperimentConfig) -> None:
    """Check complete experiment groups and actual calendar windows at the training boundary."""
    required={'experiment_id','model_name','split','pseudo_date','evaluation_end','train_end','horizon'}
    if rows.empty or not required <= set(rows):
        raise DataError('Missing experiment partition columns or empty experiment table')
    if rows.duplicated(['experiment_id','model_name']).any():
        raise DataError('Duplicate experiment/model record')
    for identifier, group in rows.groupby('experiment_id'):
        if set(group.model_name)!=set(MODELS):
            raise DataError(f'Incomplete estimator comparison: {identifier}')
        for column in ['split','pseudo_date','evaluation_end','train_end','horizon']:
            if group[column].nunique()!=1:
                raise DataError(f'Inconsistent experiment group: {identifier}: {column}')
        start=pd.Timestamp(group.pseudo_date.iloc[0]); end=pd.Timestamp(group.evaluation_end.iloc[0])
        if end!=start+pd.DateOffset(months=int(group.horizon.iloc[0])-1):
            raise DataError('Horizon does not match evaluation calendar')
        if pd.Timestamp(group.train_end.iloc[0])>=start:
            raise DataError('Training overlaps evaluation outcomes')
        expected=calendar_split(start,end,config)
        if expected=='purged' or group.split.iloc[0]!=expected:
            raise DataError('Experiment split conflicts with the configured calendar partition')
    if rows.groupby('pseudo_date').split.nunique().max()>1:
        raise DataError('Related locations/horizons split across partitions')
