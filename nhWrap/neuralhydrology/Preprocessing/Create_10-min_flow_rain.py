import pandas as pd
import os
from datetime import datetime, timedelta

# Read the main CSV file
main_df = pd.read_csv(
    r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_With_Good_Gauges_Combined_relevant.csv')


# Create output folder if it doesn't exist
# os.makedirs('flow_and_rain', exist_ok=True)


# Function to read and preprocess rain gauge data
def read_rain_gauge(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime'], dayfirst=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime'] = df['datetime'].dt.tz_localize(None)
    df.set_index('datetime', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df['Rain']


# Process each station
for _, row in main_df.iterrows():
    station_id = row['station_id']

    # Read flow data
    flow_file = f"S:/hydrolab/home/Omri_Porat/PhD/Data/Hydrological_service/Data_by_station_extrapolated_labeled/{station_id}_10_min_labeled.csv"
    flow_df = pd.read_csv(flow_file, parse_dates=['Flow_sampling_time'], dayfirst=True)
    flow_df['Flow_sampling_time'] = pd.to_datetime(flow_df['Flow_sampling_time'])
    flow_df.set_index('Flow_sampling_time', inplace=True)
    flow_df = flow_df[~flow_df.index.duplicated(keep='first')]

    # Initialize combined dataframe
    combined_df = flow_df[['Flow_m3_sec', 'Water_level_m', 'code']]

    # Process each rain gauge
    for i in range(1, 4):
        gauge_id = row[f'gauge_id_{i}']
        gauge_file = f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS/Data_by_station/Data_by_station_formatted/{gauge_id}.csv"

        if os.path.exists(gauge_file):
            rain_series = read_rain_gauge(gauge_file)
            rain_series_renamed = rain_series.rename(f'Rain_gauge_{i}')
            combined_df = combined_df.join(rain_series_renamed, how='outer')

    # Resample to ensure 10-minute intervals and forward fill
    combined_df = combined_df.resample('10min').ffill()
    combined_df.index.name = 'date'


    # Save to CSV
    output_file = f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_corrected/{station_id}_combined.csv"
    combined_df.to_csv(output_file, index=True)
    print(f"Processed station {station_id}")

print("All stations processed.")
