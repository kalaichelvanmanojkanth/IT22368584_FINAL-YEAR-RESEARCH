"""Supervised expected-error learner; reliability is not a causal probability."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = ['pre_rmse','pre_mae','trend_mismatch','residual_variance',
            'residual_autocorrelation','seasonal_residual_strength','missing_rate','horizon',
            'max_donor_weight','effective_donor_count','leave_one_donor_sensitivity',
            'model_disagreement','extrapolation_features','available_history','gap_months','placebo_error']


class ReliabilityMetaLearner:
    """Predict MAE / max(pre-period mean count, 1), with fixed boosting settings."""
    def __init__(self, seed: int = 42, features: list[str] | None = None,
                 model_kind: str = "gradient_boosting"):
        if model_kind not in {"gradient_boosting", "hist_gradient_boosting"}:
            raise ValueError("Unknown reliability model family")
        self.model_kind = model_kind
        self.seed=seed
        self.features=list(FEATURES if features is None else features)
        if not self.features or not set(self.features) <= set(FEATURES):
            raise ValueError('Feature allowlist excludes target-period outcomes, labels and identifiers')

    def _inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        required=[*self.features,'model_name']
        if not set(required)<=set(frame): raise ValueError('Missing reliability diagnostic columns')
        x=frame[required].copy()
        for column in self.features:
            x[column]=pd.to_numeric(x[column],errors='raise')
            if np.isinf(x[column]).any(): raise ValueError('Infinite diagnostic')
        if x.model_name.isna().any(): raise ValueError('Missing estimator name')
        return x

    def fit(self, experiments: pd.DataFrame):
        if experiments.empty or not experiments.split.eq('train').all():
            raise ValueError('Reliability fit accepts only the chronological training partition')
        y=experiments.target_error.to_numpy(float)
        if not np.isfinite(y).all() or (y<0).any(): raise ValueError('Invalid target error')
        x=self._inputs(experiments)
        self.available_features=[c for c in self.features if x[c].notna().any()]
        if not self.available_features: raise ValueError('No measured features')
        transform=ColumnTransformer([
            ('numeric', SimpleImputer(strategy='median',add_indicator=True), self.available_features),
            ('estimator', OneHotEncoder(handle_unknown='error',sparse_output=False), ['model_name'])])
        if self.model_kind == 'gradient_boosting':
            regressor = GradientBoostingRegressor(loss='squared_error', n_estimators=60,
                max_depth=2, min_samples_leaf=4, learning_rate=.05, random_state=self.seed)
        else:
            # Disable internal random validation/early stopping: only explicit calendar validation is legal.
            regressor = HistGradientBoostingRegressor(loss='squared_error', max_iter=60,
                max_depth=2, max_leaf_nodes=4, min_samples_leaf=4, learning_rate=.05,
                l2_regularization=1., max_bins=255, early_stopping=False,
                categorical_features=None, random_state=self.seed)
        self.pipeline=Pipeline([('features',transform),('regressor',regressor)])
        with threadpool_limits(limits=1, user_api="openmp"):
            self.pipeline.fit(x,y)
        self.reference_error=max(float(np.median(y)),1e-6)
        self.training_bounds={c:(float(x[c].min()),float(x[c].max())) for c in self.available_features}
        self.trained_until=str(pd.to_datetime(experiments.evaluation_end).max().date())
        self.training_experiment_ids=sorted(experiments.experiment_id.unique().tolist())
        return self

    def predict_expected_error(self, experiments: pd.DataFrame) -> np.ndarray:
        if not hasattr(self,'pipeline'): raise ValueError('Reliability model is not fitted')
        with threadpool_limits(limits=1, user_api="openmp"):
            return np.maximum(0.,self.pipeline.predict(self._inputs(experiments)))

    def predict_reliability(self, experiments: pd.DataFrame) -> np.ndarray:
        return 1/(1+self.predict_expected_error(experiments)/self.reference_error)

    def save(self,path: Path) -> None:
        path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)

    @classmethod
    def load(cls,path: Path):
        """Load only artifacts produced locally by this project; joblib is executable serialization."""
        obj=joblib.load(path)
        if not isinstance(obj,cls): raise ValueError('Unexpected model artifact')
        return obj


    def hyperparameters(self) -> dict:
        """Actual estimator parameters including explicitly disabled internal validation."""
        if not hasattr(self, 'pipeline'):
            raise ValueError('Reliability model is not fitted')
        return self.pipeline.named_steps['regressor'].get_params(deep=False)


class HistoricalErrorBaseline:
    """Non-ML expected error: mean TRAINING error for each counterfactual estimator."""
    def fit(self, experiments: pd.DataFrame):
        if experiments.empty or not experiments.split.eq('train').all():
            raise ValueError('Historical error baseline fits training partition only')
        values=experiments.target_error.to_numpy(float)
        if not np.isfinite(values).all() or (values<0).any():
            raise ValueError('Invalid historical error labels')
        self.errors=experiments.groupby('model_name').target_error.mean().to_dict()
        self.best_model=min(self.errors,key=self.errors.get)
        return self

    def predict_expected_error(self, experiments: pd.DataFrame) -> np.ndarray:
        result=experiments.model_name.map(self.errors)
        if result.isna().any():
            raise ValueError('Unknown estimator for historical baseline')
        return result.to_numpy(float)
