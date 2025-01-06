import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Directory containing the basin files
directory = os.path.dirname(f"S:/hydrolab/home/Omri_Porat/PhD/Data/IMS_IHS_flow_and_rain_NoBadData/")

# Data structures to store statistics and gap data
stats_list = []
gap_data = {}

# Process each file
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Basin name from filename
        basin_name = os.path.splitext(filename)[0]

        # Columns of interest
        flow_columns = ['Flow_m3_sec', 'Water_level_m']
        rain_columns = ['Rain_gauge_1', 'Rain_gauge_2', 'Rain_gauge_3']
        all_columns = flow_columns + rain_columns

        # Compute statistics
        stats = {'Basin': basin_name}
        for col in all_columns:
            if col in df.columns:
                stats.update({
                    f'{col}_min': df[col].min(skipna=True),
                    f'{col}_max': df[col].max(skipna=True),
                    f'{col}_mean': df[col].mean(skipna=True),
                    f'{col}_std': df[col].std(skipna=True),
                    f'{col}_NaNs': df[col].isna().sum(),
                    f'{col}_neg': (df[col] < 0).sum(),
                    f'{col}_gt_100': (df[col] > 100).sum(),
                    f'{col}_longest_gap': max(
                        (df[col].isna() | (df[col] < 0)).astype(int).groupby(df[col].notna().cumsum()).sum()
                    )
                })

        # Append to statistics list
        stats_list.append(stats)

        # Store data gaps for visualization
        # Identify numerical columns only
        numerical_columns = df.select_dtypes(include=[np.number]).columns

        # Compute the gap mask considering only numerical columns for negative values
        gap_mask = df.isna().any(axis=1) | (df[numerical_columns] < 0).any(axis=1)
        gap_data[basin_name] = (df['date'], gap_mask.cumsum())

# Create DataFrame of statistics
stats_df = pd.DataFrame(stats_list)


# --- Visualization Functions ---

# Heatmap
def plot_heatmap(stats_df, columns, title):
    plt.figure(figsize=(16, 11))
    data_df = stats_df.set_index('Basin')[columns]
    normalized_column_data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())
    normalized_column_data_df = normalized_column_data_df.fillna(0.5)
    _, y_params = data_df.shape
    sns.heatmap(normalized_column_data_df, annot=data_df, fmt=".2f", annot_kws={"size": 8+3*(round(8/y_params))}, cmap="coolwarm")
    plt.title(title)
    plt.xlabel('Metrics')
    plt.ylabel('Basins')
    plt.tight_layout()
    plt.show()


# Bar Plot
def plot_bar(stats_df, columns, title):
    stats_melted = stats_df.melt(id_vars='Basin', value_vars=columns, var_name='Metric', value_name='Value')
    plt.figure(figsize=(16, 11))
    sns.barplot(data=stats_melted, x='Basin', y='Value', hue='Metric')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Value')
    plt.xlabel('Basin')
    plt.tight_layout()
    plt.show()

def plot_bar_with_SD(stats_df, columns, SD_columns, title):
    stats_melted = stats_df.melt(id_vars='Basin', value_vars=columns, var_name='Metric', value_name='Value')
    stats_SD_melted = stats_df.melt(id_vars='Basin', value_vars=SD_columns, var_name='Metric', value_name='Value')
    plt.figure(figsize=(16, 11))
    sns.barplot(data=stats_melted, x='Basin', y='Value', hue='Metric')
    plt.yscale('log')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Value')
    plt.xlabel('Basin')
    plt.tight_layout()
    plt.show()

# Line Plot
def plot_line(stats_df, column, title):
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=stats_df, x='Basin', y=column, marker='o')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel('Longest Gap (10-min intervals)')
    plt.xlabel('Basin')
    plt.show()


# Data Gaps Over Time
def plot_data_gaps_over_time(gap_data):
    plt.figure(figsize=(15, 10))
    for basin_name, (dates, gaps) in gap_data.items():
        plt.plot(dates, gaps, label=basin_name, alpha=0.7)
    plt.legend(title="Basins")
    plt.title("Cumulative Data Gaps Over Time NoBadData")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Gaps")
    plt.grid(True)
    plt.show()


# --- Generate Visualizations ---

# Heatmap: Flow statistics
flow_columns = [col for col in stats_df.columns if 'Flow_m3_sec' in col]
plot_heatmap(stats_df, flow_columns, "Flow Statistics Heatmap NoBadData")

# Heatmap: Rain gauge statistics
rain_columns = [col for col in stats_df.columns if 'Rain_gauge' in col]
plot_heatmap(stats_df, rain_columns, "Rain Gauge Statistics Heatmap NoBadData")

# Bar Plot: Key Statistics - NaNs
key_columns = ['Flow_m3_sec_NaNs', 'Water_level_m_NaNs']
plot_bar(stats_df, key_columns, "NaNs by Basin NoBadData")

key_columns = ['Flow_m3_sec_mean', 'Water_level_m_mean']
SD_columns = ['Flow_m3_sec_std', 'Water_level_m_std']
plot_bar_with_SD(stats_df, key_columns, SD_columns, "Means by Basin NoBadData")

# Line Plot: Longest Data Gap
plot_line(stats_df, 'Flow_m3_sec_longest_gap', "Longest Sequence Without Data by Basin NoBadData")

# Data Gaps Over Time
plot_data_gaps_over_time(gap_data)
