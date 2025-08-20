import pandas as pd
import numpy as np
from scipy.stats import t

# Load the CSV
df = pd.read_csv("../data/rf_testing_results_all_model_dataset_1_12_2022.csv")

# Extract true values and predictions
y_true = df['Actual'].values
gep_pred = df['RF_Predicted'].values
prophet_pred = df['Prophet_Predicted'].values
lstm_pred = df['LSTM_Predicted'].values
tbats_pred = df['TBATS_Predicted'].values
chronos_pred = df['Chronos_Predicted'].values


# Define the Diebold-Mariano test
def dm_test(y_true, y1, y2, h=1, loss='mse'):
    e1 = y_true - y1
    e2 = y_true - y2

    if loss == 'mse':
        d = (e1 ** 2 - e2 ** 2)
    elif loss == 'mae':
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("Loss must be 'mse' or 'mse'")

    d_mean = np.mean(d)
    d_std = np.std(d, ddof=1)

    if d_std == 0 or np.isnan(d_std):
        print("⚠️ Zero variance in loss differential — cannot compute DM test.")
        return np.nan, np.nan

    dm_stat = d_mean / (d_std / np.sqrt(len(d)))
    p_value = 2 * t.sf(np.abs(dm_stat), df=len(d) - 1)

    return dm_stat, p_value


#Run DM tests comparing RF to other models
models = {
    "Prophet": prophet_pred,
    "LSTM": lstm_pred,
    "TBATS": tbats_pred,
    "Chronos":chronos_pred
}

for name, pred in models.items():
    stat, p_val = dm_test(y_true, gep_pred, pred, loss='mae')
    print(f"DM Test (RFvs {name}):")
    print(f"  DM Statistic = {stat:.4f}")
    print(f"  p-value = {p_val:.4f}")
    if p_val < 0.05:
        print("  → Significant difference\n")
    else:
        print("  → No significant difference\n")
