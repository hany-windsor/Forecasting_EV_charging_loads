import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "../data/boulderEV_data_01_09_2022.csv"
boulder_data = pd.read_csv(file_path)

# Convert time columns to seconds
def time_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

boulder_data['Total_Duration_s'] = boulder_data['Total_Duration__hh_mm_ss_'].apply(time_to_seconds)
boulder_data['Charging_Time_s'] = boulder_data['Charging_Time__hh_mm_ss_'].apply(time_to_seconds)


# Plot data before handling outliers
plt.figure(figsize=(12, 6))
plt.plot(boulder_data['Energy__kWh_'].reset_index(drop=True), label='Energy_KWh_(raw_data)', color='blue', alpha=0.7)
plt.title('Energy_KWh_(raw_data)')
plt.xlabel('Charging session')
plt.ylabel('Energy__kWh_')
plt.legend()
plt.grid()

# Save the plot as a PDF
plot_file = "../visualization_outputs/Energy_KWh_raw_data.pdf"
plt.savefig(plot_file, format='pdf')

# Calculate rate per minute

# Drop rows where both Energy__kWh_ and Charging_Time_s are zero
boulder_data = boulder_data[~((boulder_data['Energy__kWh_'] == 0) | (boulder_data['Charging_Time_s'] == 0))].reset_index(drop=True)

boulder_data['rate_per_min'] = boulder_data['Energy__kWh_'] / (boulder_data['Charging_Time_s'] / 60)

# Plot rate_per_min data before handling outliers
plt.figure(figsize=(12, 6))
plt.plot(boulder_data['rate_per_min'].reset_index(drop=True), label='rate_per_min(raw_data)', color='blue', alpha=0.7)
plt.title('Energy_KWh_(raw_data)')
plt.xlabel('Charging session')
plt.ylabel('Rate (kWh/min)')
plt.legend()
plt.grid()

# Save the plot as a PDF
plot_file = "../visualization_outputs/rate_per_min_raw_data.pdf"
plt.savefig(plot_file, format='pdf')



# Define outlier thresholds
acceptable_min = 0.05  # kWh per minute
acceptable_max = 0.35

# Replace outliers in Energy__kWh_ with NaN
outlier_mask = (boulder_data['rate_per_min'] < acceptable_min) | (boulder_data['rate_per_min'] > acceptable_max)
boulder_data.loc[outlier_mask, 'Energy__kWh_'] = np.nan

# Prepare features for imputation
features = ['Charging_Time_s', 'rate_per_min', 'Energy__kWh_']

# Set up the Random Forest-based imputer
rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
imputer = IterativeImputer(estimator=rf_estimator, max_iter=10, random_state=42)

# Apply imputation
boulder_data[features] = imputer.fit_transform(boulder_data[features])

output_file = "../feature_engineering_cleaned_charging_data_with_replacements.csv"
boulder_data.to_csv(output_file, index=False)


# Recalculate rate_per_min after replacing outliers
boulder_data['rate_per_min'] = boulder_data['Energy__kWh_'] / (boulder_data['Charging_Time_s'] / 60)

# Step 1: Detect outliers using Isolation Forest
features = ['Charging_Time_s', 'Energy__kWh_']
isoforest = IsolationForest(contamination=0.1, random_state=42)
boulder_data['Outlier'] = isoforest.fit_predict(boulder_data[features])

# Step 2: Replace outliers with NaN (to be imputed)
columns_to_impute = ['Charging_Time_s', 'Energy__kWh_', 'rate_per_min']
for col in columns_to_impute:
    boulder_data.loc[boulder_data['Outlier'] == -1, col] = np.nan

# Step 3: Use IterativeImputer with RandomForest to fill NaN values
rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
imputer = IterativeImputer(estimator=rf_estimator, max_iter=10, random_state=42)

boulder_data[columns_to_impute] = imputer.fit_transform(boulder_data[columns_to_impute])

# Drop Outlier flag if not needed
boulder_data.drop(columns=['Outlier'], inplace=True)

output_file = "../data/iso_forest_cleaned_charging_data_with_replacements.csv"
boulder_data.to_csv(output_file, index=False)


# Plot rate_per_min after handling outliers
plt.figure(figsize=(12, 6))
plt.plot(boulder_data['rate_per_min'].reset_index(drop=True), label='Charging Rate per Minute', color='blue', alpha=0.7)
plt.axhline(acceptable_min, color='green', linestyle='--', label='Min Acceptable Rate')
plt.axhline(acceptable_max, color='red', linestyle='--', label='Max Acceptable Rate')
plt.title('Charging Rate per Minute (Filtered and Outliers Replaced)')
plt.xlabel('Index')
plt.ylabel('Rate (kWh/min)')
plt.legend()
plt.grid()

# Save the plot as a PDF
plot_file = "../visualization_outputs/charging_rate_per_min_filtered_and_outliers_replaced.pdf"
plt.tight_layout()
plt.savefig(plot_file, format='pdf')
plt.show()


# Plot data before handling outliers
plt.figure(figsize=(12, 6))
plt.plot(boulder_data['Energy__kWh_'].reset_index(drop=True), label='Energy_KWh_(outlier_imputed)', color='blue', alpha=0.7)
plt.title('Energy_KWh_(outlier_imputed)')
plt.xlabel('Charging session')
plt.ylabel('Energy__kWh_')
plt.legend()
plt.grid()

# Save the plot as a PDF
plot_file = "../visualization_outputs/Energy_KWh_outlier_imputed.pdf"
plt.savefig(plot_file, format='pdf')