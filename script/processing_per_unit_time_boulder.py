import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the CSV file
file_path = '../data/boulderEV_data_01_09_2022.csv'
boulder_data = pd.read_csv(file_path)

# Convert the Start and End date columns to datetime
boulder_data['Start_Date___Time'] = pd.to_datetime(boulder_data['Start_Date___Time'])
boulder_data['End_Date___Time'] = pd.to_datetime(boulder_data['End_Date___Time'])

# Define the date range
start, end = "2023-06-01", "2023-08-31"
start_date = pd.to_datetime(start)
end_date = pd.to_datetime(end)

# Filter rows based on the date range
data_ranged = boulder_data[(boulder_data['Start_Date___Time'] >= start_date) & (boulder_data['End_Date___Time'] <= end_date)]

# Calculate the duration of each period in hours
data_ranged['Duration_Hours'] = (data_ranged['End_Date___Time'] - data_ranged['Start_Date___Time']).dt.total_seconds() / 3600

# Ensure no zero or negative durations to avoid division errors
data_ranged = data_ranged[data_ranged['Duration_Hours'] > 0]

# Determine the time range of the entire time horizon
start_time = data_ranged['Start_Date___Time'].min()
end_time = data_ranged['End_Date___Time'].max()


# Define the resolution of time (e.g., 1 minute)
time_resolution = pd.Timedelta(minutes=1)

# Create a DataFrame with all moments
time_range = pd.date_range(start=start_time, end=end_time, freq=time_resolution)
loads_per_unit_time = pd.DataFrame({'date': time_range})

# Initialize a column for energy in each moment
loads_per_unit_time['load'] = 0.0

# Compute the energy used in each unit of time resolution
for i, time_unit in enumerate(loads_per_unit_time['date']):
    # Find periods that include the current moment
    overlapping_periods = data_ranged[(data_ranged['Start_Date___Time'] <= time_unit) & (data_ranged['End_Date___Time'] > time_unit)]

    # Calculate the contribution of energy from each period
    energy_sum = 0
    for _, period in overlapping_periods.iterrows():
        # Ensure 'Duration_Hours' exists and is not zero
        if period['Duration_Hours'] > 0:
            energy_rate = period['Energy__kWh_'] / period['Duration_Hours']
            energy_sum += energy_rate * time_resolution.total_seconds() / 3600  # Convert seconds to hours

    # Assign the energy sum to the current time unit
    loads_per_unit_time.loc[i, 'load'] = energy_sum


file_base_name = f'{start}_{end}'.replace('/', '_')


# Aggregate loads by hour
loads_per_unit_time['date'] = loads_per_unit_time['date'].dt.floor('H')  # Floor to the nearest hour
hourly_loads = loads_per_unit_time.groupby('date')['load'].sum().reset_index()

# Save the hourly loads to a CSV
hourly_output_path = f"../data/{file_base_name}_load_per_hour_no_cleaning.csv"
hourly_loads.to_csv(hourly_output_path, index=False)
print(f"Hourly loads saved to {hourly_output_path}")

# Plot the hourly time series
hourly_pdf_filename = f"../visualization_outputs/{file_base_name}_loads_per_hour_no_cleaning.pdf"
plt.figure(figsize=(20, 12))
plt.plot(hourly_loads['date'], hourly_loads['load'], label='Loads/hour', color='orange')
plt.title('Time Series: Loads per Hour', fontsize=20, fontweight='bold')
plt.xlabel('Date and Hour', fontsize=14, fontweight='bold')
plt.ylabel('Load (kW)', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)

# Set x-axis ticks every 12 hours
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=24))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

# Rotate x-axis labels for better readability
plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=12)

plt.savefig(hourly_pdf_filename, format='pdf', bbox_inches='tight')
plt.show()

