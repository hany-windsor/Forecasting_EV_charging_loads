import pandas as pd
import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from itertools import product
import matplotlib.pyplot as plt
import time

# 🔒 Fix seeds for reproducibility
seed_value = 50
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
tf.config.experimental.enable_op_determinism()


def lstm_forecasts(input_file,date_range):
    # Load the data
    charging_loads = pd.read_csv(input_file)

    # Extract the target column
    data = charging_loads['load'].values
    data = data.reshape(-1, 1)

    # Extract the dates
    dates = charging_loads['date']

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Convert data to a supervised learning problem
    def to_supervised(data, look_back=1):
        X, y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
        return np.array(X), np.array(y)

    look_back = 3  # Use 3 timesteps to predict the next value
    X, y = to_supervised(data_scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Specify the split ratio for training and testing sets
    split_ratio = 0.8  # 80% training, 20% testing
    split_index = int(len(X) * split_ratio)

    # Split the data
    trainX, testX = X[:split_index], X[split_index:]
    trainY, testY = y[:split_index], y[split_index:]

    # Adjust training and testing dates
    train_dates = dates.iloc[look_back:split_index + look_back]  # Exclude first 'look_back' rows
    test_dates = dates.iloc[split_index + look_back:len(dates) - 1]  # Match testPredict length

    # Define the parameter grid
    param_grid = {
        'units': [100,500,1000],  # Number of neurons in the LSTM layer
        'batch_size': [10, 16, 20],  # Batch size
        'epochs': [100]  # Number of epochs
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

        # Define the LSTM model
        model = Sequential()
        model.add(LSTM(params['units'], input_shape=(look_back, 1)))
        model.add(Dense(1))  # Single output
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(trainX, trainY, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

        # Make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # Inverse scale the predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        testPredict = scaler.inverse_transform(testPredict)
        testY_actual = scaler.inverse_transform(testY.reshape(-1, 1))

        # Evaluate the model
        mse = mean_squared_error(testY_actual, testPredict)
        mae = mean_absolute_error(testY_actual, testPredict)
        mape = mean_absolute_percentage_error(testY_actual, testPredict)

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
            best_trainPredict = trainPredict
            best_testPredict = testPredict

    # Print the best parameters and results
    print("Best Parameters:")
    print(best_params)
    print(f"Best MSE: {best_mse}")

    end_training_time = time.time()

    # Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"../forecasting_results/lstm_results/lstm_grid_search_results_{date_range}.csv", index=False)


    # Save the best fitted values (training)
    trainPredict_df = pd.DataFrame({
        'date': train_dates.reset_index(drop=True),
        'fitted_values': best_trainPredict.flatten()
    })
    trainPredict_df.to_csv(f"../forecasting_results/lstm_results/LSTM_train_fitted_values_{date_range}.csv", index=False)

    # Save the best forecasted values (testing)
    testPredict_df = pd.DataFrame({
        'date': test_dates.reset_index(drop=True),
        'forecasted_values': best_testPredict.flatten()
    })
    testPredict_df.to_csv(f"../forecasting_results/lstm_results/LSTM_test_forecasted_values_{date_range}.csv", index=False)

    # Ensure datetime format for train_dates and test_dates
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)


    # Select every 12th timestamp for x-ticks
    step_hour_ticks = train_dates.iloc[::12]  # Extract every 12th timestamp

    # Calculate metrics
    train_mse_best = mean_squared_error(scaler.inverse_transform(trainY.reshape(-1, 1)), best_trainPredict)
    train_mape_best = mean_absolute_percentage_error(scaler.inverse_transform(trainY.reshape(-1, 1)), best_trainPredict)
    train_mae_best = mean_absolute_error(scaler.inverse_transform(trainY.reshape(-1, 1)), best_trainPredict)
    train_r2_best = r2_score(scaler.inverse_transform(trainY.reshape(-1, 1)), best_trainPredict)
    test_mse_best = mean_squared_error(scaler.inverse_transform(testY.reshape(-1, 1)), best_testPredict)
    test_mape_best = mean_absolute_percentage_error(scaler.inverse_transform(testY.reshape(-1, 1)), best_testPredict)
    test_mae_best = mean_absolute_error(scaler.inverse_transform(testY.reshape(-1, 1)), best_testPredict)
    test_r2_best = r2_score(scaler.inverse_transform(testY.reshape(-1, 1)), best_testPredict)


    # Define metrics
    metrics = {
        'Metric': ['MSE', 'MAPE', 'MAE', 'R2'],
        'Train': [train_mse_best, train_mape_best, train_mae_best, train_r2_best],
        'Test': [test_mse_best, test_mape_best, test_mae_best, test_r2_best]
    }

    # Create DataFrame
    metrics_df = pd.DataFrame(metrics)
    output_filename = f"../forecasting_results/lstm_results/lstm_evaluation_metrics_{date_range}.csv"
    metrics_df.to_csv(output_filename, index=False)


    # Plot 1: Fitted vs Actual for Training Set
    plt.figure(figsize=(12, 6))
    plt.plot(train_dates, scaler.inverse_transform(trainY.reshape(-1, 1)), label='Actual (Training Set)', color='blue', alpha=0.7)
    plt.plot(train_dates, best_trainPredict, label='Fitted Values', color='orange', alpha=0.7)
    plt.title(f'LSTM Model: Fitted vs Actual (Training Set): (MSE = {train_mse_best:.2f})')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    # Set x-ticks to every 12th timestamp
    #plt.xticks(ticks=step_hour_ticks.index, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/lstm_results/LSTM_fitted_vs_actual_training_set_{date_range}.pdf")


    # Select every 12th timestamp for x-ticks
    step_hour_ticks = test_dates.iloc[::12]  # Extract every 12th timestamp
    # Plot 2: Forecasted vs Actual for Testing Set
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, testY_actual, label='Actual (Testing Set)', color='blue', alpha=0.7)
    plt.plot(test_dates, best_testPredict, label='Forecasted Values', color='green', alpha=0.7)
    plt.title(f'LSTM Model: Forecasted vs Actual (Testing Set)')
    plt.xlabel('Date')
    plt.ylabel('Load')
    plt.legend()
    # Set x-ticks to every 12th timestamp
    #plt.xticks(ticks=step_hour_ticks.index, labels=step_hour_ticks.dt.strftime("%Y-%m-%d %H:%M"), rotation=90)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/lstm_results/LSTM_forecasted_vs_actual_testing_set_{date_range}.pdf")


    # Scatter plot for training data
    plt.figure(figsize=(8, 6))
    plt.scatter(scaler.inverse_transform(trainY.reshape(-1, 1)), best_trainPredict, alpha=0.6, color='blue', label='Training Data')
    #plt.plot(np.unique(trainY), np.unique(trainY), color='red', label='Ideal Fit')
    plt.title(f'LSTM-Training Set: Actual vs Fitted Values (R² = {train_r2_best:.2f})')
    plt.xlabel('Actual Load')
    plt.ylabel('Fitted Load')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/lstm_results/LSTM_training_scatter_r2_{date_range}.pdf")


    # Scatter plot for testing data
    plt.figure(figsize=(8, 6))
    plt.scatter(testY_actual, best_testPredict, alpha=0.6, color='green', label='Testing Data')
    #plt.plot(np.unique(testY_actual), np.unique(testY_actual), color='red', label='Ideal Fit')
    plt.title(f'LSTM-Testing Set: Actual vs Forecasted Values')
    plt.xlabel('Actual Load')
    plt.ylabel('Forecasted Load')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"../forecasting_results/lstm_results/LSTM_testing_scatter_r2_{date_range}.pdf")

    training_time = end_training_time - start_training_time

    return(training_time)


