import pandas as pd
from prophet import Prophet
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from itertools import product
import logging
import matplotlib.pyplot as plt
import time
import random
import numpy as np

seed = 50
random.seed(seed)
np.random.seed(seed)

# Suppress cmdstanpy logs
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def prophet_forecasts(input_file,date_range):
    # Load the data
    input_file_name = input_file
    charging_loads = pd.read_csv(input_file_name)

    # Prepare data for Prophet
    charging_loads['ds'] = pd.to_datetime(charging_loads['date'])
    charging_loads['y'] = charging_loads['load']

    # Define holidays
    cal = USFederalHolidayCalendar()
    holidays_dates = cal.holidays(start=charging_loads['ds'].min(), end=charging_loads['ds'].max())

    holidays_df = pd.DataFrame({
        'ds': holidays_dates,
        'holiday': 'USFederalHoliday'
    })

    # Split into training and testing sets
    split_ratio = 0.8
    split_index = int(len(charging_loads) * split_ratio)

    # Split the data
    train_df = charging_loads.iloc[:split_index]
    test_df = charging_loads.iloc[split_index:]

    # Ensure all date columns are in datetime format
    train_df['ds'] = pd.to_datetime(train_df['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Define the parameter grid
    param_grid = {
        #'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [1.0, 10.0, 20.0],
        'holidays_prior_scale': [0.01, 1.0, 10.0]
    }

    # Create all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]

    # Initialize variables to store the best results
    best_params = None
    best_mse = float('inf')
    results = []

    start_time = time.time()
    # Grid search
    for params in all_params:
        # Initialize Prophet model with current parameters
        model = Prophet(
            growth="flat",  # Disables the trend component
            holidays=holidays_df,
            weekly_seasonality=True,
            yearly_seasonality=True,
            #changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            uncertainty_samples = 0  # <-- deterministic outputs
        )
        model.add_seasonality(name='daily', period=24, fourier_order=5)

        # Fit the model
        model.fit(train_df[['ds', 'y']], seed=seed)

        # Create future dataframe and forecast
        prediction_length = len(test_df)
        future = model.make_future_dataframe(periods=prediction_length, freq='H')
        forecast = model.predict(future.tail(prediction_length))

        # Evaluate the model
        actual_test_values = test_df['y'].values
        predicted_test_values = forecast['yhat'].values
        mse = mean_squared_error(actual_test_values, predicted_test_values)
        mae = mean_absolute_error(actual_test_values, predicted_test_values)
        mape = mean_absolute_percentage_error(actual_test_values, predicted_test_values)

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

    # Print the best parameters and results
    print("Best Parameters:")
    print(best_params)
    print(f"Best MSE: {best_mse}")

    # Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('prophet_grid_search_results.csv', index=False)
    print("Grid search results saved to 'prophet_grid_search_results.csv'")

    # Train the best model
    best_model = Prophet(
        growth="flat",  # Disables the trend component
        holidays=holidays_df,
        weekly_seasonality=False,
        yearly_seasonality=False,
        #changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        #holidays_prior_scale=best_params['holidays_prior_scale']
    )

    best_model.fit(train_df[['ds', 'y']])

    end_time = time.time()

    # Forecast with the best model
    future = best_model.make_future_dataframe(periods=prediction_length, freq='H')
    forecast = best_model.predict(future)

    # Separate fitted values (training predictions)
    fitted_values = forecast.iloc[:len(train_df)][['ds', 'yhat']].rename(columns={'ds': 'date'})

    # Ensure datetime format for fitted values
    fitted_values['date'] = pd.to_datetime(fitted_values['date'])

    # Separate forecasted values (testing predictions)
    forecast_testing = forecast.iloc[len(train_df):][['ds', 'yhat']].rename(columns={'ds': 'date'})

    # Ensure datetime format for forecasted values
    forecast_testing['date'] = pd.to_datetime(forecast_testing['date'])

    # Calculate metrics
    train_mse_best = mean_squared_error(train_df['y'], fitted_values['yhat'])
    train_mape_best = mean_absolute_percentage_error(train_df['y'], fitted_values['yhat'])
    train_mae_best = mean_absolute_error(train_df['y'], fitted_values['yhat'])
    train_r2_best = r2_score(train_df['y'], fitted_values['yhat'])
    test_mse_best = mean_squared_error(test_df['y'], forecast_testing['yhat'])
    test_mape_best = mean_absolute_percentage_error(test_df['y'], forecast_testing['yhat'])
    test_mae_best = mean_absolute_error(test_df['y'], forecast_testing['yhat'])
    test_r2_best = r2_score(test_df['y'], forecast_testing['yhat'])

    # Define metrics
    metrics = {
        'Metric': ['MSE', 'MAPE', 'MAE', 'R2'],
        'Train': [train_mse_best, train_mape_best, train_mae_best, train_r2_best],
        'Test': [test_mse_best, test_mape_best, test_mae_best, test_r2_best]
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)

    output_filename = f"../forecasting_results/prophet_results/prophet_evaluation_metrics_{date_range}.csv"
    metrics_df.to_csv(output_filename, index=False)

    # Ensure datetime format for 'train_df' and 'test_df'
    train_df['date'] = pd.to_datetime(train_df['date'])
    test_df['date'] = pd.to_datetime(test_df['date'])

    # Plot 1: Fitted values vs Actual values for the training set
    plt.figure(figsize=(12, 6))
    plt.plot(train_df['ds'], train_df['y'], label='Actual (Training Set)', color='blue', alpha=0.6)
    plt.plot(fitted_values['date'], fitted_values['yhat'], label='Fitted Values', color='orange', alpha=0.8)
    plt.title(f'Prophet_best_model_Training Set: Actual vs Fitted Values: MSE = {train_mse_best:.2f}')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    # Select every step of timestamp for x-ticks
    step_hour_ticks = train_df["date"].iloc[::24]
    # Set x-ticks to every step in  hours
    plt.xticks(ticks=step_hour_ticks, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/prophet_results/Prophet_best_model_training_actual_vs_fitted_{date_range}.pdf")


    # Plot 2: Forecasted values vs Actual values for the testing set
    plt.figure(figsize=(12, 6))
    plt.plot(test_df['ds'], test_df['y'], label='Actual (Testing Set)', color='blue', alpha=0.6)
    plt.plot(forecast_testing['date'], forecast_testing['yhat'], label='Forecasted Values', color='green', alpha=0.8)
    plt.title(f'Prophet_best_model_Testing Set: Actual vs Forecasted Values')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    # Select every step of timestamp for x-ticks
    step_hour_ticks = test_df["date"].iloc[::12]
    # Set x-ticks to every step in  hours
    plt.xticks(ticks=step_hour_ticks, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/prophet_results/Prophet_best_model_testing_actual_vs_forecasted_{date_range}.pdf")



    # Scatter plot for training data
    plt.figure(figsize=(8, 6))
    plt.scatter(train_df['y'], fitted_values['yhat'], alpha=0.6, color='blue', label='Training Data')
    #plt.plot(np.unique(trainY), np.unique(trainY), color='red', label='Ideal Fit')
    plt.title(f'Prophet-Training Set: Actual vs Fitted Values (R² = {train_r2_best :.2f})')
    plt.xlabel('Actual Load')
    plt.ylabel('Fitted Load')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/prophet_results/Prophet_training_scatter_r2_{date_range}.pdf")


    # Scatter plot for testing data
    plt.figure(figsize=(8, 6))
    plt.scatter(test_df['y'], forecast_testing['yhat'], alpha=0.6, color='green', label='Testing Data')
    #plt.plot(np.unique(testY_actual), np.unique(testY_actual), color='red', label='Ideal Fit')
    plt.title(f'Prophet-Testing Set: Actual vs Forecasted Values')
    plt.xlabel('Actual Load')
    plt.ylabel('Forecasted Load')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/prophet_results/Prophet_testing_scatter_r2_{date_range}.pdf")



    # Save fitted values for the training set
    fitted_values.to_csv(f"../forecasting_results/prophet_results/Prophet_fitted_values_training_{date_range}.csv", index=False)

    # Save forecasted values for the testing set
    forecast_testing.to_csv(f"../forecasting_results/prophet_results/Prophet_forecasted_values_testing_{date_range}.csv", index=False)

    training_time = end_time - start_time

    return(training_time)
