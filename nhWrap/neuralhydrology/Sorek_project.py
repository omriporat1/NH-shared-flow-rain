import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv(f"S:\hydrolab\home\Omri_Porat\PhD\Classes\Basin_Project\Final_project\Cross_sections_CSV.csv")

# Step 2: Get the unique section names from the "Name" column
sections = df['Name'].unique()
begin_names = df['Begin_name'].unique()
end_names = df['End_name'].unique()

# create a "names" list of the connected end and begin names in the format "Begin_name - End_name":
names = [f"{begin} - {end}" for begin, end in zip(begin_names, end_names)]


# Step 3: Initialize variables for consistent Y-axis range
min_altitude = df['SAMPLE_1'].min()
max_altitude = df['SAMPLE_1'].max()

# Step 4: Create a single plot for all sections
plt.figure(figsize=(10, 6))  # Create the figure only once

# Step 5: Loop over each section to plot the data
for section in sections:
    # Filter the DataFrame for the current section
    section_data = df[df['Name'] == section]

    # Get the distances and altitudes for this section
    distance = section_data['distance']
    altitude = section_data['SAMPLE_1']

    # Find the index of the minimum Y value (altitude)
    min_y_index = altitude.idxmin()

    # Get the distance corresponding to the minimum Y value
    min_y_distance = distance[min_y_index]

    # Adjust the distances by subtracting the minimum Y distance to center it at distance 0
    adjusted_distance = distance - min_y_distance

    # Get the "Begin_name" and "End_name" positions
    begin_name = section_data[section_data['Name'] == 'Begin_name']
    end_name = section_data[section_data['Name'] == 'End_name']

    # Plot the data for this section
    plt.plot(adjusted_distance, altitude, label=sections)

    # Mark the Begin_name and End_name points
    # plt.scatter(begin_name['distance'], begin_name['SAMPLE_1'], color='red', label="Begin_name")
    # plt.scatter(end_name['distance'], end_name['SAMPLE_1'], color='green', label="End_name")

# Step 6: Set the Y-axis limits to be consistent across all plots
plt.ylim(min_altitude, max_altitude)
plt.axvline(x=0, linestyle='--', color='black', linewidth=1)

# Step 7: Set titles and labels
plt.xlabel("Distance [m]")
plt.ylabel("Altitude [mASL]")
# plt.legend()
# use "names" as list for legend:
plt.legend(sections)

# Step 8: Show the plot
plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd


def reduce_by_adjusting_peaks(discharge, time, target_volume):
    """
    Reduces the storm volume by lowering the highest discharge values iteratively to a consistent maximum value,
    such that the total volume is reduced to the target volume.
    """
    reduced_discharge = discharge.copy()
    while True:
        # Calculate current volume
        current_volume = calculate_storm_volume(reduced_discharge, time)
        if current_volume <= target_volume:
            break

        # Find the maximum discharge and its index
        max_discharge = reduced_discharge.max()
        max_indices = reduced_discharge[reduced_discharge == max_discharge].index

        # Reduce the maximum discharge slightly (e.g., decrement by 1% of max)
        reduction_step = 0.01 * max_discharge
        reduced_discharge.loc[max_indices] -= reduction_step

    return reduced_discharge

def calculate_storm_volume(discharge, time):
    # Convert time to seconds for proper integration
    time_seconds = (time - time.iloc[0]).dt.total_seconds()
    # Compute the integral using the trapezoidal rule
    return np.trapz(discharge, time_seconds)


# Load the uploaded hydrograph data file to inspect its structure
hydrograph_data = pd.read_csv(f"S:\hydrolab\home\Omri_Porat\PhD\Classes\Basin_Project\Final_project\18150.csv")

# Display the first few rows and basic info about the data
hydrograph_data.head(), hydrograph_data.info()

# Clean the data by retaining only relevant columns
hydrograph_data_cleaned = hydrograph_data[['Discharge (m^3/sec)', 'Time']].copy()

# Ensure the Time column is parsed correctly as datetime
hydrograph_data_cleaned['Time'] = pd.to_datetime(hydrograph_data_cleaned['Time'], errors='coerce')

# Drop rows with invalid or missing timestamps
hydrograph_data_cleaned = hydrograph_data_cleaned.dropna(subset=['Time'])
# Original storm volume

original_volume = calculate_storm_volume(
    hydrograph_data_cleaned['Discharge (m^3/sec)'],
    hydrograph_data_cleaned['Time']
)

# Reduce volume by 20%
reduction_factor = 0.2
target_volume = original_volume * (1 - reduction_factor)

# 1. Reduce every discharge value proportionally
discharge_proportional = hydrograph_data_cleaned['Discharge (m^3/sec)'] * (1 - reduction_factor)


# Apply the revised peak adjustment function
discharge_adjusted_peaks = reduce_by_adjusting_peaks(
    hydrograph_data_cleaned['Discharge (m^3/sec)'],
    hydrograph_data_cleaned['Time'],
    target_volume
)

# Add the adjusted column to the DataFrame
hydrograph_data_cleaned['Adjusted Peaks Reduction'] = discharge_adjusted_peaks

# Save the corrected dataset for inspection
output_path_corrected = '/mnt/data/modified_hydrograph_corrected.csv'
hydrograph_data_cleaned.to_csv(output_path_corrected, index=False)

output_path_corrected
