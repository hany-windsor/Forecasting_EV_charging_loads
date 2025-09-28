import pandas as pd
from tbats import TBATS
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from itertools import product
import matplotlib.pyplot as plt
import time
import random, numpy as np


seed = 42
random.seed(seed)
np.random.seed(seed)


def tbats_forecasts(input_file, date_range):
    # Load the data
    charging_loads = pd.read_csv(input_file)

    # Ensure the 'date' column is parsed as datetime
    charging_loads['date'] = pd.to_datetime(charging_loads['date'])

    # Specify the split ratio
    split_ratio = 0.8  # 80% training, 20% testing
    split_index = int(len(charging_loads) * split_ratio)

    # Define training and testing sets
    train_df = charging_loads.iloc[:split_index]
    test_df = charging_loads.iloc[split_index:]

    # Ensure only the 'load' column is passed to TBATS
    train_series = train_df['load'].values

    # Define the parameter grid for grid search
    param_grid = {
        'seasonal_periods': [(24,)],  # Daily seasonality
        'trend': [False],  # Include/exclude trend
        'damped_trend': [False],  # Include/exclude damped trend
        'use_arma_errors': [True],  # Include/exclude ARMA components
        'p': [2, 4, 6],  # AR order
        'q': [2, 4, 6],  # MA order
        'use_box_cox': ["auto"]  # Box-Cox transformation

    }

    # Create all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    # Initialize variables to store the best results
    best_params = None
    best_mse = float('inf')
    results = []

    start_training_time = time.time()

    # Grid search
    for params in all_params:
        print(f"Training with parameters: {params}")

        # Initialize TBATS model with the current parameters
        estimator = TBATS(
            seasonal_periods=params['seasonal_periods'],
            use_box_cox=params['use_box_cox'],
            use_trend=params['trend'],
            use_damped_trend=params['damped_trend'],
            use_arma_errors=params['use_arma_errors'],
            n_jobs=1  # <- avoid parallel nondeterminism
        )
        model = estimator.fit(train_series)

        # Get the fitted values of the training set
        fitted_values = model.y_hat

        # Forecast the testing set period
        forecasted_values = model.forecast(steps=len(test_df))

        # Evaluate the model
        actual_test_values = test_df['load'].values
        mse = mean_squared_error(actual_test_values, forecasted_values)
        mae = mean_absolute_error(actual_test_values, forecasted_values)
        mape = mean_absolute_percentage_error(actual_test_values, forecasted_values)

        # Store the results
        results.append({
            'params': params,
            'mse': mse,
            'mae': mae,
            'mape': mape
        })

        # Update the best model
        if mse < best_mse:
            best_mse = mse
            best_params = params
            best_model = model
            best_fitted_values = fitted_values
            best_forecasted_values = forecasted_values

    end_training_time= time.time()

    # Calculate metrics
    train_mse_best = round(mean_squared_error(train_df['load'], best_fitted_values),2)
    train_mape_best = round(mean_absolute_percentage_error(train_df['load'], best_fitted_values),2)
    train_mae_best = round(mean_absolute_error(train_df['load'], best_fitted_values),2)
    train_r2_best = round(r2_score(train_df['load'], best_fitted_values),2)
    test_mse_best = round(mean_squared_error(test_df['load'], best_forecasted_values),2)
    test_mape_best = round(mean_absolute_percentage_error(test_df['load'], best_forecasted_values),2)
    test_mae_best = round(mean_absolute_error(test_df['load'], best_forecasted_values),2)
    test_r2_best = round(r2_score(test_df['load'], best_forecasted_values),2)


    # Define metrics
    metrics = {
        'Metric': ['MSE', 'MAPE', 'MAE', 'R2'],
        'Train': [train_mse_best, train_mape_best, train_mae_best, train_r2_best],
        'Test': [test_mse_best, test_mape_best, test_mae_best, test_r2_best],
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    output_filename = f"../forecasting_results/tbats_results/tbats_evaluation_metrics_{date_range}.csv"
    metrics_df.to_csv(output_filename, index=False)

    # Save the grid search results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../forecasting_results/tbats_results/tbats_grid_search_results_{date_range}.csv", index=False)


    # Save the best fitted values (training)
    fitted_values_df = pd.DataFrame({
        'date': train_df['date'].reset_index(drop=True),
        'fitted_values': best_fitted_values
    })
    fitted_values_df.to_csv(f"../forecasting_results/tbats_results/tbats_fitting_values_{date_range}.csv", index=False)

    # Save the best forecasted values (testing)
    forecasted_values_df = pd.DataFrame({
        'date': test_df['date'].reset_index(drop=True),
        'forecasted_values': best_forecasted_values
    })
    forecasted_values_df.to_csv(f"../forecasting_results/tbats_results/tbats_forecasted_values_{date_range}.csv", index=False)

    # Plot 1: Fitted vs Actual for Training Set
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['date'], train_df['load'], label='Actual (Training Set)', color='blue', alpha=0.7)
    plt.plot(train_df['date'], best_fitted_values, label='Fitted Values', color='orange', alpha=0.7)
    plt.title(f'TBATS Model: Fitted vs Actual (Training Set)\nMSE: {train_mse_best}')
    plt.xlabel('Date')
    plt.ylabel('Load')
    step_hour_ticks = train_df['date'].iloc[::24]
    plt.xticks(ticks=step_hour_ticks, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/tbats_results/TBATS_fitted_vs_actual_training_set_{date_range}.pdf")


    # Plot 2: Forecasted vs Actual for Testing Set
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['date'], test_df['load'], label='Actual (Testing Set)', color='blue', alpha=0.7)
    plt.plot(test_df['date'], best_forecasted_values, label='Forecasted Values', color='green', alpha=0.7)
    plt.title(f'TBATS Model: Forecasted vs Actual (Testing Set)')
    plt.xlabel('Date')
    plt.ylabel('Load')
    step_hour_ticks = test_df['date'].iloc[::12]
    plt.xticks(ticks=step_hour_ticks, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/tbats_results/TBATS_forecasted_vs_actual_testing_set_{date_range}.pdf")


    # Plot 3: R² for Training Set
    plt.figure(figsize=(12, 6))
    plt.scatter(train_df['load'], best_fitted_values, alpha=0.5, color='blue')
    plt.plot([min(train_df['load']), max(train_df['load'])],
             [min(train_df['load']), max(train_df['load'])], color='red', linestyle='--')
    plt.title(f'TBATS-Training Set: R² = {train_r2_best:.2f}')
    plt.xlabel('Actual Values')
    plt.ylabel('Fitted Values')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/tbats_results/TBATS_training_r2_plot_{date_range}.pdf")


    # Plot 4: R² for Testing Set
    plt.figure(figsize=(12, 6))
    plt.scatter(test_df['load'], best_forecasted_values, alpha=0.5, color='green')
    plt.plot([min(test_df['load']), max(test_df['load'])],
             [min(test_df['load']), max(test_df['load'])], color='red', linestyle='--')
    plt.title(f'TBATS-Testing Set')
    plt.xlabel('Actual Values')
    plt.ylabel('Forecasted Values')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/tbats_results/TBATS_testing_r2_plot_{date_range}.pdf")

    training_time = end_training_time - start_training_time

    return(training_time)


