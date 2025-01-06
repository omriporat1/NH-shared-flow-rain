'''import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Folder containing your CSV files

# Lists to store analysis results
file_names = []
missing_percentages = []
negative_counts = []
high_values_counts = []

# Loop through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        data = pd.read_csv(file_path)

        # Calculate missing data percentage for available columns
        missing_data = data.isna().mean() * 100
        missing_percentages.append(missing_data)

        # Check and analyze rain gauge columns that are present
        rain_columns = ['Rain_gauge_1', 'Rain_gauge_2', 'Rain_gauge_3']
        negative_vals = []
        high_vals = []

        for col in rain_columns:
            if col in data.columns:
                negative_vals.append((data[col] < 0).sum())
                high_vals.append((data[col] > 10).sum())  # Example threshold of 10 mm
            else:
                negative_vals.append(0)  # No data for this column in the file
                high_vals.append(0)

        negative_counts.append(negative_vals)
        high_values_counts.append(high_vals)
        file_names.append(filename)

# Convert lists to DataFrames for plotting
missing_df = pd.DataFrame(missing_percentages, index=file_names).fillna(0)
neg_df = pd.DataFrame(negative_counts, index=file_names, columns=rain_columns)
high_df = pd.DataFrame(high_values_counts, index=file_names, columns=rain_columns)

# Plotting results
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

# Missing Data Heatmap
sns.heatmap(missing_df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Missing Data (%)'}, ax=axs[0],
            annot_kws={"size": 10})
axs[0].set_title("Missing Data Percentage by Column", fontsize=14)
axs[0].tick_params(axis='x', rotation=45)
axs[0].tick_params(axis='y', rotation=0)


# Function to add percentage labels on bars
def add_percentage_labels(ax, total_counts):
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height > 0:
                percentage = height / total_counts[int(bar.get_x() + bar.get_width() / 2)] * 100
                ax.annotate(f'{percentage:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)


# Negative Values Plot
total_counts_neg = neg_df.sum(axis=1)
neg_df.plot(kind='bar', stacked=True, ax=axs[1], color=['salmon', 'lightcoral', 'red'])
axs[1].set_title("Negative Values Count in Rain Gauges", fontsize=14)
axs[1].set_ylabel("Count")
add_percentage_labels(axs[1], total_counts_neg)

# High Values Plot
total_counts_high = high_df.sum(axis=1)
high_df.plot(kind='bar', stacked=True, ax=axs[2], color=['lightblue', 'deepskyblue', 'royalblue'])
axs[2].set_title("High Values Count in Rain Gauges", fontsize=14)
axs[2].set_ylabel("Count")
add_percentage_labels(axs[2], total_counts_high)

plt.xlabel("File Name")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
'''


import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set the folder path and file pattern
folder_path = f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_corrected"
file_pattern = '*.csv'
files = glob.glob(os.path.join(folder_path, file_pattern))

# Initialize lists to store statistics
missing_data_stats = []
negative_values_stats = []
high_values_stats = []

# Set thresholds
high_value_threshold = 100  # Set based on your data expectations

# Process each file
for file in files:
    # Read the CSV file
    data = pd.read_csv(file)
    file_name = os.path.basename(file)

    # Identify rain gauge columns
    rain_columns = [col for col in data.columns if any(sub in col for sub in ['Rain_gauge', 'Flow_m3_sec', 'Water'])]
    # rain_columns = [col for col in data.columns]

    # Missing data percentage per column
    missing_data = data[rain_columns].isna().mean() * 100
    missing_data_stats.append([file_name] + list(missing_data))

    # Negative values count and percentages per column
    negative_vals = [data[col][data[col] < 0].count() for col in rain_columns]
    negative_vals_percentages = [count / sum(negative_vals) * 100 if sum(negative_vals) > 0 else 0 for count in
                                 negative_vals]
    negative_values_stats.append([file_name] + list(negative_vals) + list(negative_vals_percentages))

    # High values count and percentages per column
    high_vals = [data[col][data[col] > high_value_threshold].count() for col in rain_columns]
    high_vals_percentages = [count / sum(high_vals) * 100 if sum(high_vals) > 0 else 0 for count in high_vals]
    high_values_stats.append([file_name] + list(high_vals) + list(high_vals_percentages))

# Convert lists to DataFrames for visualization
missing_df = pd.DataFrame(missing_data_stats, columns=['File'] + rain_columns)
negative_df = pd.DataFrame(negative_values_stats,
                           columns=['File'] + rain_columns + [f"{col}_percent" for col in rain_columns])
high_df = pd.DataFrame(high_values_stats, columns=['File'] + rain_columns + [f"{col}_percent" for col in rain_columns])

# Visualization 1: Missing Data Percentage Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(missing_df.set_index('File').astype(float), annot=True, cmap='YlGnBu',
            cbar_kws={'label': 'Missing Data (%)'})
plt.title('Missing Data Percentage by Column')
plt.xlabel('Rain Gauge Columns')
plt.ylabel('Files')
plt.show()

# Visualization 2: Negative Values Count with Percentages
fig, ax = plt.subplots(figsize=(12, 12))
for i, col in enumerate(rain_columns):
    ax.bar(negative_df['File'], negative_df[col], label=col, bottom=negative_df[rain_columns[:i]].sum(axis=1))

for i, col in enumerate(rain_columns):
    for idx, value in enumerate(negative_df[col]):
        if value > 0:
            percentage = negative_df[f"{col}_percent"][idx]
            ax.text(idx, negative_df[rain_columns[:i]].sum(axis=1)[idx] + value / 2, f"{percentage:.1f}%", ha='center')

plt.title('Negative Values Count in Rain Gauges with Percentage')
plt.xlabel('File Name')
plt.ylabel('Count')
plt.legend(title="Rain Gauge Columns")
plt.xticks(rotation=90)
plt.show()

# Visualization 3: High Values Count with Percentages
fig, ax = plt.subplots(figsize=(12, 12))
for i, col in enumerate(rain_columns):
    ax.bar(high_df['File'], high_df[col], label=col, bottom=high_df[rain_columns[:i]].sum(axis=1))

for i, col in enumerate(rain_columns):
    for idx, value in enumerate(high_df[col]):
        if value > 0:
            percentage = high_df[f"{col}_percent"][idx]
            ax.text(idx, high_df[rain_columns[:i]].sum(axis=1)[idx] + value / 2, f"{percentage:.1f}%", ha='center')

plt.title('High Values Count in Rain Gauges with Percentage')
plt.xlabel('File Name')
plt.ylabel('Count')
plt.legend(title="Rain Gauge Columns")
plt.xticks(rotation=45)
plt.show()


