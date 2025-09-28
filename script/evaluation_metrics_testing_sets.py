import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

# === Your exact loop ===
list_of_dates = [
    "testing_results_all_model_dataset_2022-09-01_2022-11-30.csv",
    "testing_results_all_model_dataset_2022-12-01_2023-02-28.csv",
    "testing_results_all_model_dataset_2023-03-01_2023-05-31.csv",
    "testing_results_all_model_dataset_2023-06-01_2023-08-31.csv",
]

# Where to save combined/legacy metrics
OUT_DIR = "../forecasting_results/metrics"
os.makedirs(OUT_DIR, exist_ok=True)

# New root for per-model folders (metrics + plots live under here)
MODEL_OUT_ROOT = "../forecasting_results/model_results_testing_sets"
os.makedirs(MODEL_OUT_ROOT, exist_ok=True)

TIME_CANDIDATES = ["Timestamp", "Date", "Datetime", "ds", "time"]

def compute_metrics(y_true, y_pred):
    # drop NaNs pairwise
    s = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    if s.empty:
        return np.nan, np.nan, np.nan, np.nan, 0
    y, yhat = s["y"].to_numpy(float), s["yhat"].to_numpy(float)
    mse  = mean_squared_error(y, yhat)
    mape = mean_absolute_percentage_error(y, yhat)  # fraction (0..1)
    mae  = mean_absolute_error(y, yhat)
    r2   = r2_score(y, yhat)
    return mse, mape, mae, r2, len(s)

def find_time_axis(df):
    """Return a Series for x-axis and a label. Falls back to integer index."""
    for c in TIME_CANDIDATES:
        if c in df.columns:
            x = df[c]
            # try parse to datetime if looks like date/time
            try:
                x_parsed = pd.to_datetime(x, errors="raise")
                return x_parsed, c
            except Exception:
                # not datetime—use raw values
                return x, c
    # fallback
    return pd.Series(np.arange(len(df))), "t"

def save_line_plot(model_dir, date_range, x, x_label, y_true, y_pred, model_name):
    plt.figure()
    plt.plot(x, y_true, label="Actual")
    plt.plot(x, y_pred, label="Predicted")
    plt.xlabel(x_label)
    plt.ylabel("Load")
    plt.title(f"{model_name} | Actual vs Predicted\ntesting set")
    plt.legend()

    # ⬇️ Rotate x-axis labels 90°
    ax = plt.gca()
    ax.tick_params(axis="x", labelrotation=45)

    plt.tight_layout()
    fname = os.path.join(model_dir, date_range, f"{model_name}_line_{date_range}.png")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def save_scatter_plot(model_dir, date_range, y_true, y_pred, model_name, r2):
    # drop NaNs consistently for plotting (match metrics)
    s = pd.DataFrame({"y": y_true, "yhat": y_pred}).dropna()
    if s.empty:
        return None
    y = s["y"].to_numpy(float)
    yhat = s["yhat"].to_numpy(float)

    plt.figure()
    plt.scatter(y, yhat, s=10)
    # y=x reference line
    minv = np.nanmin([y.min(), yhat.min()])
    maxv = np.nanmax([y.max(), yhat.max()])
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} | Actual vs Predicted (Scatter)\ntesting set")
    # R^2 annotation (upper left)
    # Ensure it's visible across ranges
    xpos = minv + 0.05 * (maxv - minv)
    ypos = maxv - 0.05 * (maxv - minv)
    plt.text(xpos, ypos, f"$R^2$ = {r2:.4f}")
    fname = os.path.join(model_dir, date_range, f"{model_name}_scatter_{date_range}.png")
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname

def write_model_metrics(model_dir, date_range, row_dict):
    """Write per-dataset metrics CSV inside each model's folder (appends or creates)."""
    # One file per date range so nothing overwrites
    out_csv = os.path.join(model_dir, date_range, f"{row_dict['Model']}_metrics_{date_range}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame([row_dict]).to_csv(out_csv, index=False)

def main():
    combined_rows = []

    for input_file_name in list_of_dates:
        input_filename = f"../data/{input_file_name}"  # Add path prefix
        if not os.path.exists(input_filename):
            print(f"[SKIP] Not found: {input_filename}")
            continue

        # date_range from name
        date_range = os.path.splitext(
            input_file_name.replace("testing_results_all_model_dataset_", "")
        )[0]

        # Load the file
        df = pd.read_csv(input_filename)

        actual_col = "Actual"

        if actual_col not in df.columns:
            raise KeyError(
                f"'Actual' column not found in {input_filename}. "
                f"Available columns: {list(df.columns)}"
            )

        # Prediction columns: prefer those ending with '_Predicted'
        pred_cols = [c for c in df.columns if c.endswith("_Predicted")]

        # Find x-axis (time/index)
        x, x_label = find_time_axis(df)

        rows = []
        for c in pred_cols:
            model_name = c.replace("_Predicted", "")

            # Metrics
            mse, mape, mae, r2, n = compute_metrics(df[actual_col], df[c])
            row = {
                "DateRange": date_range,
                "Model": model_name,
                "MSE": mse,
                "MAPE": mape,
                "MAE": mae,
                "R2": r2,
                "N": n,
            }
            rows.append(row)

            # Per-model folder
            model_dir = os.path.join(MODEL_OUT_ROOT, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # === Visualizations ===
            try:
                # 1) Actual vs Predicted line chart
                save_line_plot(model_dir, date_range, x, x_label, df[actual_col], df[c], model_name)

                # 2) Scatter (Actual vs Predicted) with R^2
                save_scatter_plot(model_dir, date_range, df[actual_col], df[c], model_name, r2)
            except Exception as e:
                print(f"[WARN] Plotting failed for {model_name} on {date_range}: {e}")

            # Write per-dataset metrics inside the model's folder
            try:
                write_model_metrics(model_dir, date_range, row)
            except Exception as e:
                print(f"[WARN] Writing per-model metrics failed for {model_name} on {date_range}: {e}")

        # Save per-file metrics next to OUT_DIR (legacy/combined)
        metrics_df = pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)
        out_path = os.path.join(OUT_DIR, f"model_metrics_{date_range}")
        metrics_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

        combined_rows.append(metrics_df)

    # Combined across all date ranges
    if combined_rows:
        all_df = pd.concat(combined_rows, ignore_index=True)
        all_out = os.path.join(OUT_DIR, "model_metrics_ALL_DATES.csv")
        all_df.to_csv(all_out, index=False)
        print(f"Saved combined metrics: {all_out}")
    else:
        print("No metrics produced. Check paths and column names.")

if __name__ == "__main__":
    main()
