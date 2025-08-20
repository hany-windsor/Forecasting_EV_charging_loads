import pandas as pd
import numpy as np

def adding_context(ensemble_data):
    # Convert the 'date' column to datetime
    ensemble_data['date'] = pd.to_datetime(ensemble_data['date'])

    # Extract time-based features
    ensemble_data['hour'] = ensemble_data['date'].dt.hour              # Hour of the day (0–23)
    #ensemble_data['day_of_week'] = ensemble_data['date'].dt.dayofweek  # Day of the week (Monday=0, Sunday=6)
    #ensemble_data['month'] = ensemble_data['date'].dt.month            # Month of the year (1–12)

    # Compute cumulative MSE for each forecaster
    #for col in ['yhat_lstm', 'yhat_prophet', 'yhat_tbats']:
    #    errors = (ensemble_data['actual'] - ensemble_data[col]) ** 2
    #    ensemble_data[f'cumulative_mse_{col}'] = np.cumsum(errors) / (np.arange(1, len(errors) + 1))

    # Compute per-timestep absolute errors
    for col in ['yhat_lstm', 'yhat_prophet', 'yhat_tbats']:
        ensemble_data[f'error_{col}'] = np.abs(ensemble_data['actual'] - ensemble_data[col])


    # Rank the absolute errors at each timestep (row-wise), lower error = better rank (1 = best)
    error_cols = ['error_yhat_lstm', 'error_yhat_prophet', 'error_yhat_tbats']
    ranks = ensemble_data[error_cols].rank(axis=1, method='min', ascending=True)

    # Assign ranks to new columns
    ensemble_data['rank_lstm'] = ranks['error_yhat_lstm']
    ensemble_data['rank_prophet'] = ranks['error_yhat_prophet']
    ensemble_data['rank_tbats'] = ranks['error_yhat_tbats']

    return ensemble_data
