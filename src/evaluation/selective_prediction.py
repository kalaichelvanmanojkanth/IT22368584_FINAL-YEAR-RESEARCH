"""Validation-derived selection with explicit limitations and fail-closed evidence checks."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class EvidencePolicy:
    predicted_error_limit: float
    disagreement_limit: float
    pre_rmse_limit: float
    extrapolation_limit: float
    reference_error: float
    max_seen_gap: float
    min_horizon: int
    max_horizon: int
    min_calibration_blocks: int
    reliability_threshold: float = 0.0

    @classmethod
    def fit(cls, validation: pd.DataFrame, quantile: float, reference_error: float, minimum_blocks: int):
        if validation.empty or not validation.split.eq('validation').all(): raise ValueError('Policy must use validation only')
        return cls(*[float(validation[c].quantile(quantile)) for c in
                     ['predicted_error','model_disagreement','pre_rmse','extrapolation_features']],
                   reference_error,float(validation.gap_months.max()),int(validation.horizon.min()),
                   int(validation.horizon.max()),minimum_blocks)

    def assess(self, diagnostics: dict, calibration_blocks: int) -> dict:
        reasons=[]; hard=[]
        error=diagnostics.get('predicted_error')
        if error is None or not np.isfinite(error):
            return {'decision':'INSUFFICIENT_EVIDENCE','reliability_score':None,'reasons':['No valid predicted error']}
        score=float(1/(1+error/max(self.reference_error,1e-6)))
        if score < getattr(self, 'reliability_threshold', 0.):
            hard.append('Model Reliability Score below validation-selected threshold')
        if diagnostics.get('missing_rate',0)>0: hard.append('Missing surveillance outcomes; fitted model is outside complete-data training support')
        if diagnostics.get('gap_months',0)>self.max_seen_gap: hard.append('Unvalidated gap between training history and intervention window')
        if not self.min_horizon<=diagnostics.get('horizon',0)<=self.max_horizon: hard.append('Evaluation horizon outside validation support')
        for field,limit,label in [('predicted_error',self.predicted_error_limit,'High predicted error'),
                                 ('model_disagreement',self.disagreement_limit,'Strong estimator disagreement'),
                                 ('pre_rmse',self.pre_rmse_limit,'Poor pre-intervention fit'),
                                 ('extrapolation_features',self.extrapolation_limit,'Control diagnostics outside typical validation range')]:
            if diagnostics.get(field,0)>limit: reasons.append(label)
        if calibration_blocks<self.min_calibration_blocks: reasons.append('Too few independent calendar calibration blocks')
        if diagnostics.get('effective_donor_count',2)<=1: reasons.append('Single donor; donor-removal stability cannot be established')
        decision='INSUFFICIENT_EVIDENCE' if hard else ('UNCERTAIN_REVIEW_REQUIRED' if reasons else 'SUPPORTED_ESTIMATE')
        return {'decision':decision,'reliability_score':score,'reasons':hard+reasons}


def risk_coverage(errors, scores) -> pd.DataFrame:
    errors,scores=np.asarray(errors,float),np.asarray(scores,float)
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(scores).all(): raise ValueError('Finite nonempty inputs required')
    rows=[]
    for threshold in sorted(set([0.,*scores.tolist()])):
        retained=scores>=threshold
        rows.append({'threshold':threshold,'coverage':float(retained.mean()),'retained_error':float(errors[retained].mean()),'retained_count':int(retained.sum())})
    return pd.DataFrame(rows)


@dataclass
class ValidationSelector:
    """Choose lowest validation retained risk subject to a prespecified coverage floor.

    Ties prefer greater coverage. Rejection uses scores only; target errors are never
    accepted by the deployed accept() method. This is exploratory threshold selection,
    not a statistical guarantee that test retained risk will decrease.
    """
    threshold: float
    minimum_coverage: float
    validation_coverage: float
    validation_risk: float
    validation_experiment_ids: list[str]

    @classmethod
    def fit(cls, validation: pd.DataFrame, minimum_coverage: float = .5):
        if validation.empty or not validation.split.eq('validation').all():
            raise ValueError('Selective threshold selection accepts validation experiments only')
        if validation.experiment_id.duplicated().any():
            raise ValueError('Threshold selection requires one ensemble row per experiment')
        if not 0 < minimum_coverage <= 1:
            raise ValueError('Invalid minimum retained coverage')
        curve = risk_coverage(validation.normalized_mae, validation.reliability_score)
        eligible = curve[curve.coverage >= minimum_coverage]
        chosen = eligible.sort_values(['retained_error','coverage','threshold'],
                                      ascending=[True,False,True]).iloc[0]
        return cls(float(chosen.threshold), minimum_coverage, float(chosen.coverage),
                   float(chosen.retained_error), sorted(validation.experiment_id.tolist()))

    def accept(self, scores) -> np.ndarray:
        values = np.asarray(scores, float)
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError('Reliability scores must be finite in [0,1]')
        return values >= self.threshold

    def evaluate(self, experiments: pd.DataFrame) -> dict:
        retained = self.accept(experiments.reliability_score)
        return {'threshold':self.threshold,'analyses':len(experiments),
                'retained_count':int(retained.sum()),'coverage':float(retained.mean()),
                'all_error':float(experiments.normalized_mae.mean()),
                'retained_error':float(experiments.loc[retained,'normalized_mae'].mean()) if retained.any() else None}
