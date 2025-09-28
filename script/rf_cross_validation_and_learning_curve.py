import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import re


input_filename = "../data/ensemble_data_with_context_2022-09-01_2022-11-30.csv"

ensemble_data= pd.read_csv(input_filename)

date_range = re.search(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", input_filename).group()

X_data = ensemble_data.drop(columns=['actual', 'date'])
y_data = ensemble_data['actual']
tscv = TimeSeriesSplit(n_splits=10)
model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)

train_r_sqaure_scores = []
test_r_sqaure_scores = []
train_mse_scores = []
test_mse_scores = []
train_sizes = []
test_sizes = []

# Define the parameter grid
param_values= {
    'n_estimators': 1000,
    'max_depth': 5,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 50
}

# Initialize the model with the specified parameters
rf_model = RandomForestRegressor(**param_values)

for train_index, test_index in tscv.split(X_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

    rf_model.fit(X_train, y_train)
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    train_r_sqaure_scores.append(train_r2)
    test_r_sqaure_scores.append(test_r2)
    train_sizes.append(len(train_index))
    test_sizes = len(test_index)
    train_mse_scores.append(mse_train)
    test_mse_scores.append(mse_test)

# Plot Learning Curve (R²)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_r_sqaure_scores, label='Train R²', marker='o')
plt.plot(train_sizes, test_r_sqaure_scores, label='Test R²', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.title(f'Random_Forest_Learning_Curve_R_square_{date_range}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"../forecasting_results/random_forest_results/Random_Forest_Learning_Curve_R_square_{date_range}.png"
plt.savefig(plot_filename)
plt.show()

# Plot Learning Curve (mse)
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mse_scores, label='MSE_training', marker='o')
plt.plot(train_sizes, test_mse_scores, label='MSE_testing', marker='o')
plt.xlabel('Training Set Size')
plt.ylabel('MSE')
plt.title(f'Random_Forest_Learning_Curve_mse_{date_range}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plot_filename = f"../forecasting_results/random_forest_results/Random_Forest_Learning_Curve_mse_{date_range}.png"
plt.savefig(plot_filename)
plt.show()


# Create the DataFrame
learning_curve_df = pd.DataFrame({
    'Train Size': train_sizes,
    'Test Size': test_sizes,
    'Train Score R_square': train_r_sqaure_scores,
    'Test Score R_square': test_r_sqaure_scores,
    'Train MSE':mse_train,
    'Test MSE':mse_test
})

# Optionally, save to CSV
learning_curve_df.to_csv(f"../forecasting_results/random_forest_results/RF_learning_curve_results_{date_range}.csv", index=False)
