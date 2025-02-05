import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



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
        reduction_step = 0.005 * max_discharge
        reduced_discharge.loc[max_indices] -= reduction_step

    return reduced_discharge

def calculate_storm_volume(discharge, time):
    # Convert time to seconds for proper integration
    time_seconds = (time - time.iloc[0]).dt.total_seconds()
    # Compute the integral using the trapezoidal rule
    return np.trapz(discharge, time_seconds)


# Load the uploaded hydrograph data file to inspect its structure
hydrograph_data = pd.read_csv(r"S:\hydrolab\home\Omri_Porat\PhD\Classes\Basin_Project\Final_project\a18150.csv")

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
reduction_factors = [0.2, 0.5, 0.8]


for reduction_factor in reduction_factors:
    target_volume = original_volume * (1 - reduction_factor)

    # Reduce every discharge value proportionally
    discharge_proportional = hydrograph_data_cleaned['Discharge (m^3/sec)'] * (1 - reduction_factor)

    # Apply the revised peak adjustment function
    discharge_adjusted_peaks = reduce_by_adjusting_peaks(
        hydrograph_data_cleaned['Discharge (m^3/sec)'],
        hydrograph_data_cleaned['Time'],
        target_volume
    )

    # Add the adjusted column to the DataFrame
    hydrograph_data_cleaned[f'Adjusted Peaks Reduction {reduction_factor * 100}%'] = discharge_adjusted_peaks
    hydrograph_data_cleaned[f'Adjusted Factor Reduction {reduction_factor * 100}%'] = discharge_proportional

    # calculate the adjusted volume:
    adjusted_volume = calculate_storm_volume(discharge_adjusted_peaks, hydrograph_data_cleaned['Time'])

# save the corrected dataset for inspection:
output_path_corrected = r"S:\hydrolab\home\Omri_Porat\PhD\Classes\Basin_Project\Final_project\a18150_adjusted.csv"
hydrograph_data_cleaned.to_csv(output_path_corrected, index=False)

# Plot the original and adjusted discharge values:
colors = ['r', 'g', 'b']
plt.figure(figsize=(10, 7))
plt.plot(hydrograph_data_cleaned['Time'], hydrograph_data_cleaned['Discharge (m^3/sec)'], label='Original', color='#FFAD00')


for reduction_factor, color in zip(reduction_factors, colors):
    plt.plot(
        hydrograph_data_cleaned['Time'],
        hydrograph_data_cleaned[f'Adjusted Peaks Reduction {reduction_factor * 100}%'],
        label=f'Adjusted Peaks {reduction_factor * 100}%',
        linestyle='-.',
        color=color
    )
    plt.plot(
        hydrograph_data_cleaned['Time'],
        hydrograph_data_cleaned[f'Adjusted Factor Reduction {reduction_factor * 100}%'],
        label=f'Adjusted Factor {reduction_factor * 100}%',
        linestyle='--',
        color=color
    )
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))  # Format: 30-Jan-2013
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Show every 1 day

plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Discharge (m$^3$/sec)')
plt.title('Original and Adjusted Discharge Values')
plt.legend()
plt.subplots_adjust(bottom=0.2)
plt.show()

# save the new figure:
output_path_figure = r"S:\hydrolab\home\Omri_Porat\PhD\Classes\Basin_Project\Final_project\a18150_adjusted.png"
plt.savefig(output_path_figure)

