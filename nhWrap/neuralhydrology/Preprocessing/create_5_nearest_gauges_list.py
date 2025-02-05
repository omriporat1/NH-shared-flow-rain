import pandas as pd

# Load the CSV file
# file_path = r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_5NearetsGauges.csv'
file_path = r'S:\hydrolab\home\Omri_Porat\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\Preprocessing\Centroids_5_Nearest_Good_Gauges1.csv'
df = pd.read_csv(file_path)

# Initialize a list to hold the transformed data
rows = []

# Iterate through each unique station
for station_id, group in df.groupby('station_id'):
    # Initialize a dictionary to hold data for the current station
    row_data = {
        'station_id': station_id,
        'station_x': group['station_x'].iloc[0],
        'station_y': group['station_y'].iloc[0]
    }

    # Iterate through the gauges associated with the current station
    for i, gauge in enumerate(group.itertuples(), start=1):
        row_data[f'gauge_id_{i}'] = gauge.gauge_id
        row_data[f'gauge_name_{i}'] = gauge.gauge_name
        row_data[f'n_{i}'] = gauge.n
        row_data[f'distance_{i}'] = gauge.distance
        row_data[f'gauge_x_{i}'] = gauge.gauge_x
        row_data[f'gauge_y_{i}'] = gauge.gauge_y

    # Append the row data to the list
    rows.append(row_data)

# Convert the list of dictionaries to a DataFrame
transformed_data = pd.DataFrame(rows)

# Save the transformed data to a new CSV file
output_path = r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_With_Good_Gauges_Combined.csv'
transformed_data.to_csv(output_path, index=False)

print(f"File saved to {output_path}")
