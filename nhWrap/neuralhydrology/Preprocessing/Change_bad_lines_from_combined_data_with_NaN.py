# import relevant:
import os
import pandas as pd
import numpy as np

origin_directory = os.path.dirname(f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_corrected/")
output_directory = os.path.dirname(f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_BadToNaN/")

columns_of_interest = ['Flow_m3_sec', 'Water_level_m', 'Rain_gauge_1', 'Rain_gauge_2', 'Rain_gauge_3']
# read each csv file in the origin directory and replace all lines where there is NaN or negative values or values
# above 400 with NaN for all columns of interest:
for filename in os.listdir(origin_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(origin_directory, filename)
        df = pd.read_csv(file_path)
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Columns of interest
        for col in columns_of_interest:
            if col in df.columns:
                # Replace invalid rows
                mask_invalid = (df[col].isna()) | (df[col] < 0) | (df[col] > 400)
                df.loc[mask_invalid, columns_of_interest] = np.nan

        # save the new csv file
        output_path = os.path.join(output_directory, filename)
        df.to_csv(output_path, index=False)
        print(f"File saved to {output_path}")
