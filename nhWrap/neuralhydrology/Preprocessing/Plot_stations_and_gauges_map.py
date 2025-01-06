'''import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
from shapely.geometry import Point


stations_df = pd.read_csv(r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Data_by_station_extrapolated_labeled\shared_model_IHS_stations.csv')
gauges_df = pd.read_csv(r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_With_Gauges_Combined.csv')

# Print information about the original dataframes
print("Stations DataFrame:")
print(stations_df.columns)
print(stations_df.head())

print("\nGauges DataFrame:")
print(gauges_df.columns)
print(gauges_df.head())

# Convert Station_ID and station_id to strings
stations_df['Station_ID'] = stations_df['Station_ID'].astype(str)
gauges_df['station_id'] = gauges_df['station_id'].str.split('_').str[-1]

# Merge the dataframes
merged_df = pd.merge(stations_df, gauges_df, left_on='Station_ID', right_on='station_id', how='left')

# Print information about the merged dataframe
print("\nMerged DataFrame:")
print(merged_df.columns)
print(merged_df[['Station_ID', 'station_x', 'station_y', 'gauge_x_1', 'gauge_y_1']].head())

# Load Israel shapefile
israel = gpd.read_file(r'S:\hydrolab\home\Omri_Porat\PhD\Data\gadm41_ISR_shp\gadm41_ISR_0.shp')

# Convert to ITM projection (EPSG:2039)
israel = israel.to_crs(epsg=2039)

# Create points from coordinates
merged_df['geometry'] = [Point(xy) for xy in zip(merged_df['station_x'], merged_df['station_y'])]

# Convert merged_df to GeoDataFrame
gdf = gpd.GeoDataFrame(merged_df, crs="EPSG:2039", geometry='geometry')

# Create the plot
fig, ax = plt.subplots(figsize=(20, 30))

# Plot Israel's outline
israel.boundary.plot(ax=ax, color='gray', linewidth=0.5)

# Plot stations
gdf.plot(ax=ax, color='red', markersize=50, zorder=3)

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])

# Calculate the maximum distance for color scaling
max_distance = gdf[['distance_1', 'distance_2', 'distance_3']].max().max()

# Plot stations and gauges
for _, row in gdf.iterrows():
    # Annotate the station
    ax.annotate(row['Station_ID'], (row['station_x'], row['station_y']),
                xytext=(3, 3), textcoords='offset points', fontsize=8, zorder=4)

    # Plot the gauges and lines
    for i in range(1, 4):
        gauge_x = row[f'gauge_x_{i}']
        gauge_y = row[f'gauge_y_{i}']
        distance = row[f'distance_{i}']

        if pd.notna(gauge_x) and pd.notna(gauge_y):
            ax.scatter(gauge_x, gauge_y, color='blue', s=20, zorder=3)

            line = ax.plot([row['station_x'], gauge_x], [row['station_y'], gauge_y], '-',
                           linewidth=1, alpha=0.5, zorder=2,
                           color=cmap(distance / max_distance))

ax.set_title("Stations and Related Gauges in Israel", fontsize=20)
ax.set_xlabel('X Coordinate (ITM)', fontsize=14)
ax.set_ylabel('Y Coordinate (ITM)', fontsize=14)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_distance))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Distance', fontsize=14)

# Add legend
ax.scatter([], [], color='red', s=50, label='Stations')
ax.scatter([], [], color='blue', s=20, label='Gauges')
ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
plt.savefig("all_stations_and_gauges_map.png", dpi=300, bbox_inches='tight')
print("Map has been created and saved as 'all_stations_and_gauges_map.png'.")
'''


import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point
import branca.colormap as cm

# Load data
stations_df = pd.read_csv(r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Data_by_station_extrapolated_labeled\shared_model_IHS_stations.csv')
gauges_df = pd.read_csv(r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_With_Good_Gauges_Combined.csv')

# Convert Station_ID and station_id to strings
stations_df['Station_ID'] = stations_df['Station_ID'].astype(str)
gauges_df['station_id'] = gauges_df['station_id'].str.split('_').str[-1]

# Merge the dataframes
merged_df = pd.merge(stations_df, gauges_df, left_on='Station_ID', right_on='station_id', how='left')

# export the merged data to a new CSV file:
output_path = r'S:\hydrolab\home\Omri_Porat\PhD\Data\Hydrological_service\Stations_With_Gauges_Combined_relevant1.csv'
merged_df.to_csv(output_path, index=False)

# Load Israel shapefile
israel = gpd.read_file(r'S:\hydrolab\home\Omri_Porat\PhD\Data\gadm41_ISR_shp\gadm41_ISR_0.shp')

# Convert to WGS84 projection (EPSG:4326) for use with Folium
israel = israel.to_crs(epsg=4326)

# Create points from coordinates and convert to WGS84
merged_df['geometry'] = [Point(xy) for xy in zip(merged_df['station_x'], merged_df['station_y'])]
gdf = gpd.GeoDataFrame(merged_df, crs="EPSG:2039", geometry='geometry')
gdf = gdf.to_crs(epsg=4326)

# Calculate the center of Israel for the map
center_lat, center_lon = israel.geometry.centroid.y.mean(), israel.geometry.centroid.x.mean()

# Create a map
m = folium.Map(location=[center_lat, center_lon], zoom_start=8)

# Add Israel's outline
folium.GeoJson(israel).add_to(m)

# Create a colormap for distances
max_distance = gdf[['distance_1', 'distance_2', 'distance_3']].max().max()
colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=0, vmax=max_distance)

# Create a MarkerCluster for stations
station_cluster = MarkerCluster(name="Stations").add_to(m)

# Add stations and gauges to the map
for _, row in gdf.iterrows():
    # Add station marker
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=f"Station ID: {row['Station_ID']}",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(station_cluster)

    # Add gauges and lines
    for i in range(1, 4):
        gauge_x = row[f'gauge_x_{i}']
        gauge_y = row[f'gauge_y_{i}']
        distance = row[f'distance_{i}']

        if pd.notna(gauge_x) and pd.notna(gauge_y):
            # Convert gauge coordinates to WGS84
            gauge_point = gpd.GeoDataFrame(geometry=[Point(gauge_x, gauge_y)], crs="EPSG:2039").to_crs(epsg=4326)
            gauge_lat, gauge_lon = gauge_point.geometry.y[0], gauge_point.geometry.x[0]

            # Add gauge marker
            folium.Marker(
                location=[gauge_lat, gauge_lon],
                popup=f"Gauge {i} for Station {row['Station_ID']}",
                icon=folium.Icon(color='blue', icon='tint')
            ).add_to(m)

            # Add line between station and gauge
            folium.PolyLine(
                locations=[[row.geometry.y, row.geometry.x], [gauge_lat, gauge_lon]],
                color=colormap(distance),
                weight=2,
                opacity=0.8
            ).add_to(m)

# Add color legend
colormap.add_to(m)
colormap.caption = 'Distance'

# Add layer control
folium.LayerControl().add_to(m)

# Save the map
m.save("interactive_stations_and_good_gauges_map.html")
print("Interactive map has been created and saved as 'interactive_stations_and_good_gauges_map.html'.")
