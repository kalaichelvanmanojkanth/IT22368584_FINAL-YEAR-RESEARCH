"""Empirical trajectory-error calibration preserving whole calendar dependence blocks.

This is not an exact conformal guarantee. Maxima over overlapping windows and all
areas within each calendar year are kept together; sparse years imply weak evidence.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BlockResidualCalibration:
    quantile: float
    block_count: int
    alpha: float
    scores: list[float]

    @classmethod
    def fit(cls, trajectories: pd.DataFrame, alpha: float=.1):
        if trajectories.empty or not trajectories.split.eq('calibration').all():
            raise ValueError('Intervals use calibration partition only')
        frame=trajectories.copy()
        frame['block']=pd.to_datetime(frame.date).dt.year
        frame['score']=np.abs(frame.observed-frame.prediction)/frame.outcome_scale
        scores=frame.groupby('block').score.max().to_numpy(float)
        if not np.isfinite(scores).all(): raise ValueError('Invalid calibration residuals')
        q=float(np.quantile(scores,1-alpha,method='higher'))
        return cls(q,len(scores),alpha,scores.tolist())

    def interval(self,prediction,scale: float):
        prediction=np.asarray(prediction,float)
        if scale<=0 or not np.isfinite(scale): raise ValueError('Invalid historical outcome scale')
        width=self.quantile*scale
        return np.maximum(0.,prediction-width),prediction+width

    def effect_interval(self,observed,prediction,scale: float):
        lower,upper=self.interval(prediction,scale)
        return float(np.sum(observed)-np.sum(upper)),float(np.sum(observed)-np.sum(lower))
