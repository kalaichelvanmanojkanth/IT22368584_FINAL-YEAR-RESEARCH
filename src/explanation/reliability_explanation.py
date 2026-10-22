"""Expose the learned error signals and explicit evidence-policy reasons."""


def explain(diagnostics: dict, assessment: dict, bundle: dict) -> dict:
    return {'label':'Model Reliability Score',
            'definition':'1 / (1 + predicted normalized MAE / median training normalized MAE)',
            'interpretation':'Relative predicted model error; not a probability of causality or clinical risk',
            'factors':{k:diagnostics.get(k) for k in ['predicted_error','pre_rmse','model_disagreement',
                    'missing_rate','extrapolation_features','gap_months','effective_donor_count']},
            'decision':assessment['decision'],'reasons':assessment['reasons'],
            'calibration_calendar_years':bundle['calibration'].block_count,
            'unavailable_signals':['Single-donor removal stability','Validated real-world causal ground truth'],
            'interval_warning':'Empirical calendar-block intervals; exact coverage is not established'}
