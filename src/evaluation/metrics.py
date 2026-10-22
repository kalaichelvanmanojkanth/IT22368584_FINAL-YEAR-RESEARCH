"""Prediction metrics; post-intervention factual outcomes are not Y(0) labels."""
import numpy as np


def prediction_metrics(observed: np.ndarray, predicted: np.ndarray,
                       training: np.ndarray | None = None, season: int = 12) -> dict[str, float | None]:
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    if observed.shape != predicted.shape or not observed.size:
        raise ValueError("Metric vectors must have matching, nonempty shapes")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics cannot silently discard nonfinite outcomes")
    errors = observed - predicted
    result = {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.mean(errors ** 2))), "mase": None}
    if training is not None and len(training) > season:
        scale = float(np.abs(np.asarray(training)[season:] - np.asarray(training)[:-season]).mean())
        if np.isfinite(scale) and scale > 1e-12:
            result["mase"] = result["mae"] / scale
    return result


def effect_summary(observed: np.ndarray, counterfactual: np.ndarray) -> dict[str, float | None]:
    """Signed factual-minus-counterfactual contrast; negative indicates fewer cases."""
    prediction_metrics(observed, counterfactual)  # boundary checks, not an accuracy claim
    difference = float(np.sum(observed) - np.sum(counterfactual))
    denominator = float(np.sum(counterfactual))
    return {"effect_absolute": difference, "estimated_cases_averted": -difference,
            "effect_percent": 100 * difference / denominator if denominator > 1e-8 else None}
