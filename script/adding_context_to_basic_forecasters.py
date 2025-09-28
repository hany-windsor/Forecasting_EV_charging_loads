import pandas as pd
import numpy as np

def adding_context(ensemble_data: pd.DataFrame, lag_hours: int = 24) -> pd.DataFrame:
    ensemble_data_with_context = ensemble_data.copy()

    # Ensure datetime + chronological order
    ensemble_data_with_context['date'] = pd.to_datetime(ensemble_data_with_context['date'])
    ensemble_data_with_context = ensemble_data_with_context.sort_values('date').reset_index(drop=True)

    # Time feature
    ensemble_data_with_context['hour'] = ensemble_data_with_context['date'].dt.hour

    models = ['yhat_lstm', 'yhat_prophet', 'yhat_tbats']

    # Compute errors and lagged errors
    for m in models:
        err = (ensemble_data_with_context['actual'] - ensemble_data_with_context[m]).abs()
        ensemble_data_with_context[f'error_{m}'] = err                                  # same-timestamp (diagnostic only)
        ensemble_data_with_context[f'error_{m}_lag{lag_hours}'] = err.shift(lag_hours)   # safe, uses past only

    # Rank by lagged errors (lower error = better rank)
    lag_err_cols = [f'error_{m}_lag{lag_hours}' for m in models]
    ranks = ensemble_data_with_context[lag_err_cols].rank(axis=1, method='min', ascending=True)
    ranks.columns = ['rank_lstm', 'rank_prophet', 'rank_tbats']
    ensemble_data_with_context[['rank_lstm', 'rank_prophet', 'rank_tbats']] = ranks

    # Drop the same-timestamp error columns
    cols_to_drop = [f'error_{m}' for m in models]  # error_yhat_lstm, error_yhat_prophet, error_yhat_tbats
    ensemble_data_with_context.drop(columns=[c for c in cols_to_drop if c in ensemble_data_with_context.columns], inplace=True)

    # Drop all rows that contain any NaNs (e.g., first `lag_hours` rows)
    ensemble_data_with_context = ensemble_data_with_context.dropna().reset_index(drop=True)

    return ensemble_data_with_context
