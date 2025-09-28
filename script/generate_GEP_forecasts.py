import pandas as pd
import re
import json
from script import GEP_ensemble_forecast, adding_context_to_basic_forecasters

list_of_dates = ["2022-09-01_2022-11-30_load_per_hour.csv","2022-12-01_2023-02-28_load_per_hour.csv","2023-03-01_2023-05-31_load_per_hour.csv", "2023-06-01_2023-08-31_load_per_hour.csv"]

training_time = {}

for input_file_name in list_of_dates:
    input_filename = f"../data/{input_file_name}"  # Add path prefix
    charging_loads = pd.read_csv(input_filename)

    # Extract the date range
    date_range = re.search(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", input_filename).group()

    # Split into training and testing sets
    split_ratio = 0.8
    split_index = int(len(charging_loads) * split_ratio)

    # Split the data
    train_data = charging_loads.iloc[:split_index]
    test_data = charging_loads.iloc[split_index:]

    # Load forecasted data (training and testing)
    lstm_forecast_training = pd.read_csv(f"../forecasting_results/lstm_results/LSTM_train_fitted_values_{date_range}.csv")
    prophet_forecast_training = pd.read_csv(f"../forecasting_results/prophet_results/Prophet_fitted_values_training_{date_range}.csv")
    tbats_forecast_training = pd.read_csv(f"../forecasting_results/tbats_results/tbats_fitting_values_{date_range}.csv")

    lstm_forecast_testing = pd.read_csv(f"../forecasting_results/lstm_results/LSTM_test_forecasted_values_{date_range}.csv")
    prophet_forecast_testing = pd.read_csv(f"../forecasting_results/prophet_results/Prophet_forecasted_values_testing_{date_range}.csv")
    tbats_forecast_testing = pd.read_csv(f"../forecasting_results/tbats_results/tbats_forecasted_values_{date_range}.csv")

    # Parse dates
    train_data['date'] = pd.to_datetime(train_data['date'])
    lstm_forecast_training['date'] = pd.to_datetime(lstm_forecast_training['date'])
    prophet_forecast_training['date'] = pd.to_datetime(prophet_forecast_training['date'])
    tbats_forecast_training['date'] = pd.to_datetime(tbats_forecast_training['date'])

    test_data['date'] = pd.to_datetime(test_data['date'])
    lstm_forecast_testing['date'] = pd.to_datetime(lstm_forecast_testing['date'])
    prophet_forecast_testing['date'] = pd.to_datetime(prophet_forecast_testing['date'])
    tbats_forecast_testing['date'] = pd.to_datetime(tbats_forecast_testing['date'])

    # Merge training set with forecasts
    ensemble_data_training = train_data[['date', 'load']].rename(columns={'load': 'actual'})
    lstm_forecast_training = lstm_forecast_training.rename(columns={'fitted_values': 'yhat_lstm'})
    prophet_forecast_training = prophet_forecast_training.rename(columns={'yhat': 'yhat_prophet'})
    tbats_forecast_training = tbats_forecast_training.rename(columns={'fitted_values': 'yhat_tbats'})

    ensemble_data_training = pd.merge(ensemble_data_training, lstm_forecast_training[['date', 'yhat_lstm']], on='date',
                                      how='inner')
    ensemble_data_training = pd.merge(ensemble_data_training, prophet_forecast_training[['date', 'yhat_prophet']],
                                      on='date', how='inner')
    ensemble_data_training = pd.merge(ensemble_data_training, tbats_forecast_training[['date', 'yhat_tbats']], on='date',
                                      how='inner')

    # adding context and cumulative MSE to the training set
    ensemble_data_training_with_context = adding_context_to_basic_forecasters.adding_context(ensemble_data_training)

    # Merge testing set with forecasts
    ensemble_data_testing = test_data[['date', 'load']].rename(columns={'load': 'actual'})
    lstm_forecast_testing = lstm_forecast_testing.rename(columns={'forecasted_values': 'yhat_lstm'})
    prophet_forecast_testing = prophet_forecast_testing.rename(columns={'yhat': 'yhat_prophet'})
    tbats_forecast_testing = tbats_forecast_testing.rename(columns={'forecasted_values': 'yhat_tbats'})

    ensemble_data_testing = pd.merge(ensemble_data_testing, lstm_forecast_testing[['date', 'yhat_lstm']], on='date',
                                     how='inner')
    ensemble_data_testing = pd.merge(ensemble_data_testing, prophet_forecast_testing[['date', 'yhat_prophet']], on='date',
                                     how='inner')
    ensemble_data_testing = pd.merge(ensemble_data_testing, tbats_forecast_testing[['date', 'yhat_tbats']], on='date',
                                     how='inner')
    # adding context and cumulative MSE to the testing set
    ensemble_data_testing_with_context = adding_context_to_basic_forecasters.adding_context(ensemble_data_testing)

    column_list = ensemble_data_training_with_context.columns.tolist()


    # run ensemble forecasts
    ensemble_composition = "lstm_prophet_tbats"
    if ensemble_composition not in training_time:
        training_time[ensemble_composition] = {}
    training_time_of_one_model = GEP_ensemble_forecast.GEP_ensemble(date_range, ensemble_composition,
                                                                    ensemble_data_training_with_context,
                                                                    ensemble_data_testing_with_context)
    training_time[ensemble_composition][date_range] = training_time_of_one_model


# Save to JSON file
with open("../forecasting_results/GEP_ensemble_forecasts/training_time.json", "w") as json_file:
 json.dump(training_time, json_file, indent=4)