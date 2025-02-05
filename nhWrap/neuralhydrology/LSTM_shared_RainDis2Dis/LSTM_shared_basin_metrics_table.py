import pandas as pd
import openpyxl
import plotly.graph_objects as go
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'run_metrics_by_basin_ensemble.xlsx'
data = pd.ExcelFile(file_path)

# Parse the relevant sheet
sheet_name = 'run_metrics_by_basin'
df = data.parse(sheet_name)

# Prepare a table where the columns represent basins, and include basin ID and name in the header

# Filter reference and ensemble rows
reference_df = df[df['run_title'] == 'Reference']
ensemble_df = df[df['run_title'] != 'Reference']

# Initialize the result dictionary to store the final table
result = {}

# Loop through unique basins to build the columns
for basin_id, basin_name in zip(reference_df['basin'], reference_df['basin_name']):
    # Get reference NSE and RMSE
    ref_row = reference_df[reference_df['basin'] == basin_id]
    ref_nse = ref_row['NSE'].values[0]
    ref_rmse = ref_row['RMSE'].values[0]

    # Get ensemble NSE and RMSE stats (mean, min, max)
    ens_rows = ensemble_df[ensemble_df['basin'] == basin_id]
    ens_nse_mean = ens_rows['NSE'].mean()
    ens_nse_median = ens_rows['NSE'].median()
    ens_nse_min = ens_rows['NSE'].min()
    ens_nse_max = ens_rows['NSE'].max()
    ens_rmse_mean = ens_rows['RMSE'].mean()
    ens_rmse_median = ens_rows['RMSE'].median()
    ens_rmse_min = ens_rows['RMSE'].min()
    ens_rmse_max = ens_rows['RMSE'].max()

    # Create column data
    result[f"{basin_id} - {basin_name}"] = [
        f"NSE: {ref_nse:.2f}, RMSE: {ref_rmse:.2f}",
        f"NSE: {ens_nse_median:.2f} ({ens_nse_min:.2f}, {ens_nse_max:.2f}), "
        f"RMSE: {ens_rmse_median:.2f} ({ens_rmse_min:.2f}, {ens_rmse_max:.2f})"
    ]

# Convert the result to a DataFrame
result_df = pd.DataFrame(result, index=["Reference", "Ensemble - median (min, max)"])

# Save the result to a CSV file if needed
result_df.to_csv('basin_metrics_table1.csv', index_label="Metric")

# Display the result
print(result_df)

'''

# Split the reference and ensemble rows into separate DataFrames
reference = result_df.loc["Reference"].str.extract(r'NSE: ([\d.-]+), RMSE: ([\d.-]+)').astype(float)
ensemble = result_df.loc["Ensemble"].str.extract(r'NSE: ([\d.-]+) \(([\d.-]+), ([\d.-]+)\), RMSE: ([\d.-]+) \(([\d.-]+), ([\d.-]+)\)').astype(float)

# Combine data for visualization
heatmap_data = pd.DataFrame({
    'Reference NSE': reference[0],
    'Reference RMSE': reference[1],
    'Ensemble NSE (Mean)': ensemble[0],
    'Ensemble RMSE (Mean)': ensemble[3]
}, index=result_df.columns)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("Basin Metrics Heatmap")
plt.ylabel("Basins")
plt.xlabel("Metrics")
plt.show()



# Prepare data for bar plot
x = np.arange(len(result_df.columns))  # X-axis positions
width = 0.35  # Width of bars

# Extract metrics for plotting
ref_nse = reference[0]
ens_nse = ensemble[0]
ref_rmse = reference[1]
ens_rmse = ensemble[3]

# Create the plot
plt.figure(figsize=(14, 8))
plt.bar(x - width/2, ref_nse, width, label='Reference NSE')
plt.bar(x + width/2, ens_nse, width, label='Ensemble NSE (Mean)')

# Customize the chart
plt.xticks(x, result_df.columns, rotation=45, ha='right', fontsize=9)
plt.title("Reference vs. Ensemble NSE Across Basins")
plt.ylabel("NSE Value")
plt.legend()
plt.tight_layout()
plt.show()

# You can repeat the same for RMSE by swapping `ref_nse` and `ens_nse` with `ref_rmse` and `ens_rmse`.
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=result_df.values,
                 colLabels=result_df.columns,
                 rowLabels=result_df.index,
                 loc='center',
                 cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(result_df.columns))))
plt.title("Basin Metrics Table")
plt.show()



fig = go.Figure(data=[go.Table(
    header=dict(values=['Metric'] + list(result_df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[result_df.index] + [result_df[col].values for col in result_df.columns],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(title="Basin Metrics Table")
fig.show()
'''
