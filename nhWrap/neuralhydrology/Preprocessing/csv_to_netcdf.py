import os
import pandas as pd
import xarray as xr

# Define the paths
input_folder = f"C:/PhD/Data/Caravan/timeseries/csv/archive/il_before_2024_12_24"
attribute_file = f"C:/PhD/Data/Caravan/attributes/il/attributes_other_il.csv"
output_folder_csv = f"C:/PhD/Data/Caravan/timeseries/csv/il"
output_folder_netcdf = f"C:/PhD/Data/Caravan/timeseries/netcdf/il"


# Ensure the output folder exists
os.makedirs(output_folder_netcdf, exist_ok=True)
os.makedirs(output_folder_csv, exist_ok=True)

# create dict to store normalization values - min and max for each feature of each basin:
norm_dict = {}


# Loop through each CSV file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        csv_path = os.path.join(input_folder, file_name)
        # Extract the numeric ID (XXXX) from the file name
        numeric_id = file_name.split('_')[0]  # Assumes the format is XXXX_combined.csv
        basin_id = 'il_' + numeric_id

        df = pd.read_csv(csv_path, na_values=['', ' '], parse_dates=['date'], dayfirst=True)

        # Convert the date column to datetime and set it as the index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # extract basin area from attribute_file:
        attribute_df = pd.read_csv(attribute_file)
        basin_area = attribute_df.loc[attribute_df['gauge_id'] == basin_id, 'area'].values[0]

        norm_dict[basin_id] = {
            "basin_area": basin_area,
            "features": {}
        }

        # normalize each of the feature between [0,1]:
        for feature in df.columns:
            if feature == 'date':
                continue
            min_val = df[feature].min()
            max_val = df[feature].max()

            # add a new column with the normalized values named like the original + _minmax_norm:
            df[f"{feature}_minmax_norm"] = (df[feature] - min_val) / (max_val - min_val)

            # Save min and max to the dictionary under the basin ID
            norm_dict[basin_id]["features"][feature] = {
                "min": min_val,
                "max": max_val
            }
            if feature == 'Flow_m3_sec':
                df['unit_discharge_m3_sec_km'] = df[feature] / basin_area


        # Convert the DataFrame to an xarray Dataset
        ds = xr.Dataset.from_dataframe(df)



        # Define the output path for the netCDF file
        netcdf_file_name = f"il_{numeric_id}.nc"
        netcdf_path = os.path.join(output_folder_netcdf, netcdf_file_name)
        # Save the xarray Dataset as a netCDF file
        ds.to_netcdf(netcdf_path)

        # Define the output path for the CSV file
        csv_file_name = f"il_{numeric_id}.csv"
        csv_path = os.path.join(output_folder_csv, csv_file_name)
        # Save the DataFrame as a CSV file
        df.to_csv(csv_path)

        print(f"Converted {file_name} to {netcdf_file_name}")
