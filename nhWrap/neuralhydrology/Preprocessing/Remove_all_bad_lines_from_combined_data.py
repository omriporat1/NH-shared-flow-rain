# import relevant:
import os
import pandas as pd
import numpy as np
import seaborn as sns

origin_directory = os.path.dirname(f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_corrected/")
output_directory = os.path.dirname(f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_NoBadData/")

columns_of_interest = ['Flow_m3_sec', 'Water_level_m', 'Rain_gauge_1', 'Rain_gauge_2', 'Rain_gauge_3']
# read each csv file in the origin directory and remove all lines where there is NaN or negative values or values
# above 400 in the columns of interest:
for filename in os.listdir(origin_directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(origin_directory, filename)
        df = pd.read_csv(file_path)
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Columns of interest
        all_columns = columns_of_interest
        # Compute statistics
        for col in all_columns:
            if col in df.columns:
                # remove all lines where there is NaN or negative values or values above 400 in the columns of interest:
                df = df.dropna(subset=[col])
                df = df[df[col] >= 0]
                df = df[df[col] <= 400]
        # save the new csv file
        output_path = os.path.join(output_directory, filename)
        df.to_csv(output_path, index=False)
        print(f"File saved to {output_path}")
