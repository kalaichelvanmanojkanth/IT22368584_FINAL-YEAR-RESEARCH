"""Nonnegative model weighting from predicted normalized expected errors."""
import numpy as np
from scipy.special import softmax


def error_weights(errors, temperature: float = 1., strategy: str = 'softmax', epsilon: float = 1e-8):
    values=np.asarray(errors,float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError('Expected errors must be a finite nonnegative vector')
    if temperature<=0 or epsilon<=0: raise ValueError('Positive temperature/epsilon required')
    if strategy=='softmax': return softmax(-values/temperature)
    if strategy=='inverse':
        inverse=1/np.maximum(values,epsilon); return inverse/inverse.sum()
    raise ValueError('Unknown weighting strategy')


def combine(predictions,weights):
    values=np.asarray(predictions,float); weights=np.asarray(weights,float)
    if values.ndim!=2 or values.shape[0]!=len(weights) or not np.isfinite(values).all(): raise ValueError('Invalid trajectories')
    if not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.sum(),1): raise ValueError('Invalid normalized weights')
    return weights@values
