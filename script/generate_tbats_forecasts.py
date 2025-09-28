import model_tbats_with_grid_search
import re
import pandas as pd

if __name__ == "__main__":

    # Runs on cleaned data (outlier detection and imputation)
    list_of_dates = ["2022-09-01_2022-11-30_load_per_hour.csv","2022-12-01_2023-02-28_load_per_hour.csv","2023-03-01_2023-05-31_load_per_hour.csv", "2023-06-01_2023-08-31_load_per_hour.csv"]

    # Runs on uncleaned data
    #list_of_dates = ["2022-09-01_2022-11-30_load_per_hour_no_cleaning.csv","2022-12-01_2023-02-28_load_per_hour_no_cleaning.csv","2023-03-01_2023-05-31_load_per_hour_no_cleaning.csv","2023-06-01_2023-08-31_load_per_hour_no_cleaning.csv"]

    training_time = {}
    for input_file_name in list_of_dates:
        input_filename = f"../data/{input_file_name}"  # Add path prefix
        date_range = re.search(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", input_file_name).group()
        training_time_of_one_model = model_tbats_with_grid_search.tbats_forecasts(input_filename, date_range)
        training_time[date_range] = training_time_of_one_model

    training_time_df = pd.DataFrame(list(training_time.items()), columns=['Date', 'Training_Time'])
    training_time_df.to_csv("../forecasting_results/tbats_results/training_time.csv", index=False)
