"""Diagnostics computed exclusively from historical fitted residuals."""
import numpy as np
from src.evaluation.metrics import prediction_metrics


def pre_diagnostics(observed: np.ndarray, fitted: np.ndarray) -> dict:
    residuals = np.asarray(observed) - np.asarray(fitted)
    variance = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else None
    autocorrelation = None
    if len(residuals) > 2 and np.std(residuals[:-1]) > 1e-12 and np.std(residuals[1:]) > 1e-12:
        autocorrelation = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    return {"pre_fit_metrics": prediction_metrics(observed, fitted, observed),
            "residual_variance": variance, "residual_autocorrelation": autocorrelation,
            "available_history": len(residuals), "missing_rate": 0.0,
            "residuals": residuals.tolist()}


def reliability_features(model, pre, controls, dates, missing_rate: float = 0.) -> dict:
    """Scale-aware diagnostics using pre outcomes and allowed post controls only."""
    d = model.diagnostics()
    scale = max(float(np.mean(model.training_observed)), 1.)
    residual = np.asarray(d['residuals']) / scale
    target = model.training_observed / scale
    donor_pre = model.donor_matrix(pre[pre.location_id.isin(model.control_units)], model.pre_dates,
                                    model.control_units, 'dengue_incidence').mean(axis=1)
    donor_post = model.donor_matrix(controls, dates, model.control_units, 'dengue_incidence').mean(axis=1)
    donor_scale = max(float(np.std(donor_pre)), 1.)
    trend = float(np.polyfit(np.arange(len(target)), target, 1)[0])
    donor_trend = float(np.polyfit(np.arange(len(donor_pre)), donor_pre / max(donor_pre.mean(), 1.), 1)[0])
    seasonal = np.array([residual[np.asarray(model.pre_dates.month) == month].mean() for month in range(1,13)])
    gap = (dates[0].year - model.pre_dates[-1].year) * 12 + dates[0].month - model.pre_dates[-1].month - 1
    return {'pre_rmse': d['pre_fit_metrics']['rmse'] / scale,
            'pre_mae': d['pre_fit_metrics']['mae'] / scale,
            'trend_mismatch': abs(trend - donor_trend),
            'residual_variance': float(np.var(residual)),
            'residual_autocorrelation': d['residual_autocorrelation'],
            'seasonal_residual_strength': float(np.var(seasonal)),
            'missing_rate': missing_rate, 'horizon': len(dates),
            'max_donor_weight': d.get('max_donor_weight'),
            'effective_donor_count': d.get('effective_donor_count'),
            'leave_one_donor_sensitivity': None, 'prediction_interval_width': None,
            'placebo_error': None,
            'extrapolation_features': float(np.max(np.abs(donor_post - donor_pre.mean())) / donor_scale),
            'available_history': len(target), 'gap_months': gap, 'outcome_scale': scale,
            'model_disagreement': None}
