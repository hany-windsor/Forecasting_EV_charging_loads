import os
import re
import json
import numpy as np
import pandas as pd
from scipy.stats import t

# Files to loop over (examples)
list_of_dates = [
    "testing_results_all_model_dataset_2022-09-01_2022-11-30.csv",
    "testing_results_all_model_dataset_2022-12-01_2023-02-28.csv",
    "testing_results_all_model_dataset_2023-03-01_2023-05-31.csv",
    "testing_results_all_model_dataset_2023-06-01_2023-08-31.csv",
]

def build_path(name: str) -> str:
    return f"../data/{name}"

def get_date_range(name: str) -> str:
    """
    Extract YYYY-MM-DD_YYYY-MM-DD from a file name.
    Falls back to the stem if no match is found.
    """
    m = re.search(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", name)
    if m:
        return m.group(0)
    # fallback: drop extension & directories
    return os.path.splitext(os.path.basename(name))[0]

def dm_test(y_true, y1, y2):
    e1 = y_true - y1
    e2 = y_true - y2

    d = (e1**2 - e2**2)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return (float('nan'), float('nan'))
    d_mean = np.mean(d)
    d_std  = np.std(d, ddof=1)
    if d_std == 0 or not np.isfinite(d_std):
        return (float('nan'), float('nan'))
    dm_stat = d_mean / (d_std / np.sqrt(n))
    p_value = 2 * t.sf(np.abs(dm_stat), df=n - 1)
    return (float(dm_stat), float(p_value))

def find_column(df, name):
    if name in df.columns:
        return name
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    raise KeyError(f"Column '{name}' not found in: {list(df.columns)}")

def main(alpha: float = 0.05):
    out_dir = "../forecasting_results/metrics"
    os.makedirs(out_dir, exist_ok=True)

    all_results_json = {"alpha": alpha, "files": {}}
    all_rows_csv = []

    for input_file_name in list_of_dates:
        input_filename = build_path(input_file_name)  # keep .csv to read
        if not os.path.exists(input_filename):
            print(f"[SKIP] Not found: {input_filename}")
            continue

        date_tag = get_date_range(input_file_name)  # <-- only the date range
        df = pd.read_csv(input_filename)

        actual_col = find_column(df, "Actual")
        rf_col     = find_column(df, "RF_Predicted")
        gep_col    = find_column(df, "GEP_Predicted")

        candidate_models = {
            "Prophet": "Prophet_Predicted",
            "LSTM": "LSTM_Predicted",
            "TBATS": "TBATS_Predicted",
            "Chronos": "Chronos_Predicted",
            "RF": "RF_Predicted",
            "GEP": "GEP_Predicted",
        }
        available = {k: v for k, v in candidate_models.items() if v in df.columns}

        y_true  = pd.to_numeric(df[actual_col], errors="coerce").to_numpy()
        rf_pred = pd.to_numeric(df[rf_col], errors="coerce").to_numpy()
        gep_pred= pd.to_numeric(df[gep_col], errors="coerce").to_numpy()

        # models for comparison (exclude RF & GEP themselves)
        others = {}
        for name, col in available.items():
            if name in ("RF", "GEP"):
                continue
            others[name] = pd.to_numeric(df[col], errors="coerce").to_numpy()

        # Group 1: RF vs all except GEP
        group1 = {}
        per_file_rows = []
        for name, pred in others.items():
            stat, pval = dm_test(y_true, rf_pred, pred)
            sig = (pval < alpha) if np.isfinite(pval) else False
            group1[name] = {"dm_stat": stat, "p_value": pval, "significant": sig}
            per_file_rows.append({
                "Dataset": date_tag,
                "Ensemble_model": "RF",
                "CompareTo": name,
                "dm_stat": stat,
                "p_value": pval,
                "significant": sig,
                "n_obs": int(np.sum(np.isfinite(y_true))),
            })

        # Group 2: GEP vs all except RF
        group2 = {}
        for name, pred in others.items():
            stat, pval = dm_test(y_true, gep_pred, pred)
            sig = (pval < alpha) if np.isfinite(pval) else False
            group2[name] = {"dm_stat": stat, "p_value": pval, "significant": sig}
            per_file_rows.append({
                "Dataset": date_tag,
                "Ensemble_model": "GEP",
                "CompareTo": name,
                "dm_stat": stat,
                "p_value": pval,
                "significant": sig,
                "n_obs": int(np.sum(np.isfinite(y_true))),
            })

        # Save per-file CSV using only the date range
        per_file_df = pd.DataFrame(per_file_rows)
        per_file_csv = os.path.join(out_dir, f"dm_test_two_groups_{date_tag}.csv")
        per_file_df.to_csv(per_file_csv, index=False)
        print(f"Saved per-file CSV: {per_file_csv}")

        all_rows_csv.append(per_file_df)

        # JSON keyed by date range only
        all_results_json["files"][date_tag] = {
            "n_obs": int(np.sum(np.isfinite(y_true))),
            "RF_significance": group1,
            "GEP_significance": group2,
        }

    # Combined CSV across all files
    if all_rows_csv:
        combined_df = pd.concat(all_rows_csv, ignore_index=True)
        combined_csv = os.path.join(out_dir, "dm_test_results.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"Saved combined CSV: {combined_csv}")

    # JSON
    out_json = os.path.join(out_dir, "dm_test_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results_json, f, indent=2)
    print(f"Saved JSON: {out_json}")

if __name__ == "__main__":
    main()
