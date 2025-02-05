# imports:

import requests
import json
from pyproj import Transformer
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import random
import pandas as pd
from datetime import timedelta
import matplotlib.colors as mcolors


# Shower properties:
d_shower = 10 # Shower duration (minutes)
H_shower = "20:00"  # Shower time (HH:mm, 24h)
t_d = 40  # Minimal desired shower temperature (°C)
water_dis = 8.0  # Flow rate during the shower (l/min)
heating_d_ref = 40  # duration of heating in winter (minutes) - before usage

# Location (relevant IMS station)
station = 22 # Jerusalem Givat Ram


# Tank and collector properties:
V = 100  # Water tank volume (liters)
m = V  # Water tank mass (Kg)
A_c = 2.0 # Area of collecting panel
eff = 0.7 # efficiency of the collector (%)
A_t = 1.74 # surface area of the water tank (m^2)
dt = 3600
U_t = 5
c_p = 4186
P_elec = 3000
eff_elec = 1.0
heat_loss_coefficient = 2 # Coefficient representing the rate of heat loss from the tank
T_cap = 70 # Safety limitation for tank temperature (°C)

# Scientific constants:
density = 1000 # density of water (Kg/m^3)
specific_heat = 4186 # Specific heat of water (J/kg °C)
t_room = 20 # room temperature (°C)



def calculate_shower_temp_with_plot(t_d, d_shower, V, water_dis=6, t_room=25, is_plot=False, is_print=False):
    """
    Calculate the minimal initial tank temperature for a shower of given duration and temperature,
    and optionally plot the tank temperature, discharge, and consumption temperature.

    Args:
        t_d (float): Desired water temperature during the shower (°C).
        d_shower (float): Duration of the shower (minutes).
        water_dis (float): Flow rate during the shower (liters per minute).
        V (float): Tank volume (liters).
        t_room (float): Room temperature (°C, fresh water temperature).
        plot (bool): Whether to plot the tank temperature, discharge, and consumption temperature.
        is_print (bool): Whether to print debug information.
    Returns:
        float: Minimal initial tank temperature (°C).
    """
    # Initialize variables
    time_steps = int(d_shower)  # Shower duration in minutes
    tank_temp = t_d  # Final tank temperature after the last minute

    # Lists for plotting
    minutes = []
    tank_temps = []
    discharge_temps = []
    tank_discharge_rates = []

    if is_print:
        # Print initial conditions
        print(f"Initial Conditions: Desired Shower Temp: {t_d}°C, Shower Duration: {d_shower} min, "
              f"Flow Rate: {water_dis} L/min, Tank Volume: {V} L, Room Temp: {t_room}°C\n")

    # Iterate backward from the last minute of the shower
    for minute in range(time_steps, 0, -1):
        # Calculate mix ratio
        mix_ratio = (t_d - t_room) / (tank_temp - t_room)
        if mix_ratio > 1:
            raise ValueError(f"Tank water temperature is too low at minute {minute} to maintain the desired shower temperature.")

        # Tank water discharge rate (based on the mix ratio)
        tank_discharge_rate = water_dis * mix_ratio

        # Record data for plotting
        minutes.append(minute)
        tank_temps.append(tank_temp)
        discharge_temps.append(t_d)
        tank_discharge_rates.append(tank_discharge_rate)

        # Calculate the required initial tank temperature for the current step
        tank_temp = (tank_temp * V - t_room * tank_discharge_rate) / (V - tank_discharge_rate)

        if is_print:
            # Print debug info
            print(f"Minute {minute}: Tank Temp: {tank_temp:.2f}°C, Mix Ratio: {mix_ratio:.2f}, Discharge Rate: {tank_discharge_rate:.2f} L/min")

    # Reverse the lists for proper plotting (time progresses forward)
    minutes.reverse()
    tank_temps.reverse()
    discharge_temps.reverse()
    tank_discharge_rates.reverse()

    if is_print:
        # The resulting tank_temp is the required initial temperature
        print(f"\nMinimal Initial Tank Temperature: {tank_temp:.2f}°C\n")

    # Plot the results if requested
    if is_plot:
        plt.figure(figsize=(12, 6))

        # Plot tank temperature
        plt.subplot(2, 1, 1)
        plt.plot(minutes, tank_temps, label="Tank Temperature", color="blue")
        plt.axhline(y=t_d, color="red", linestyle="--", label="Desired Shower Temp")
        plt.axhline(y=t_room, color="green", linestyle="--", label="Room Temp (Fresh Water)")
        plt.ylabel("Temperature (°C)")
        plt.title("Tank Temperature Over Time")
        plt.legend()

        # Plot tank water discharge rate
        plt.subplot(2, 1, 2)
        plt.plot(minutes, tank_discharge_rates, label="Tank Water Discharge Rate", color="purple")
        plt.ylabel("Discharge Rate (L/min)")
        plt.title("Tank Water Discharge Rate Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return tank_temp


# Calculate minimal initial tank temperature and plot the results
try:
    min_tank_temp = calculate_shower_temp_with_plot(t_d, d_shower, V, water_dis, t_room, is_print=True, is_plot=True)
    print(f"Minimal Initial Tank Temperature: {min_tank_temp:.2f}°C")
except ValueError as e:
    print(f"Error: {e}")

# Extract the relevant channels (Temp, Global radiation) for Jerusalem Givat Ram station and then retrieves the data from midnight to the current time

export_folder = '/content/drive/My Drive/Classes/AI_for_social_good/Project/code/'
csv_filename = 'available_stations_boiler.csv'
full_path = export_folder + csv_filename

url = "https://api.ims.gov.il/v1/Envista/stations"

headers = {
    'Authorization': 'ApiToken 6d2dd889-3fcf-4987-986f-e4679d4b2400'
}

response = requests.request("Get", url, headers=headers)
stations_data = json.loads(response.text.encode('utf8'))
transformer = Transformer.from_crs("epsg:4326", "epsg:2039", always_xy=True)

station_list = []

for current_station in stations_data:
    is_station_active = current_station["active"]
    is_has_location = isinstance(current_station["location"]["longitude"], (int, float))
    is_has_grad = any("rad" in monitor.get("name", "") for monitor in current_station.get("monitors", []))
    is_has_time = any("Time" in monitor.get("name", "") for monitor in current_station.get("monitors", []))
    is_correct_station = current_station["stationId"] == station
    if is_station_active and is_has_location and is_has_grad and is_has_time and is_correct_station:

        lon, lat = current_station["location"]["longitude"], current_station["location"]["latitude"]
        # print(lat, lon)

        for monitor in current_station["monitors"]:
            if monitor["name"] == "Grad":
                Grad_channelID = monitor["channelId"]
            elif monitor["name"] == "Time":
                Time_channelID = monitor["channelId"]
            elif monitor["name"] == "TD":
                Temp_channelID = monitor["channelId"]

now = datetime(2025, 1, 22, 13, 0)
yesterday = (now - timedelta(days=1)).strftime("%Y/%m/%d")
today = now.strftime("%Y/%m/%d")
tomorrow = (now + timedelta(days=1)).strftime("%Y/%m/%d")

date_range = 'from=' + today + '&to=' + tomorrow


def fetch_channel_data(station, channel, date_range, headers):
    url = "https://api.ims.gov.il/v1/Envista/stations/" + station + '/data/' + channel + '/?' + date_range
    response = requests.request("get", url, headers=headers)

    if response.text.strip():  # Check if the response content is not empty
        single_channel_data = json.loads(response.text.encode('utf8'))
        return [
            {"datetime": entry["datetime"], "value": entry["channels"][0]["value"]}
            for entry in single_channel_data["data"]
        ]
    else:
        print(f"Empty response for channel {channel} from the API.")
        return []


# Fetch data for both channels
data_Grad = fetch_channel_data(str(station), str(Grad_channelID), date_range, headers)
data_Temp = fetch_channel_data(str(station), str(Temp_channelID), date_range, headers)

# retrieve sunrise time for station lat, lon:
sunrise_api_url = "https://api.sunrise-sunset.org/json"

# API request parameters
sunrise_params = {
    "lat": lat,
    "lng": lon,
    "date": today,
    "formatted": 0,  # Use ISO 8601 format
    "tzid": "Asia/Jerusalem"
}

# Make the API request
response = requests.get(sunrise_api_url, params=sunrise_params)
if response.status_code == 200:
    sunrise = response.json()['results']['sunrise']
    print(f"Sunrise time (Israel local time): {sunrise}")
else:
    print("Error fetching sunrise time.")

# Combine data from both channels
if data_Grad and data_Temp:
    data_Grad = pd.DataFrame(data_Grad).rename(columns={"value": "Grad"})
    data_Temp = pd.DataFrame(data_Temp).rename(columns={"value": "TD"})

    # Merge the data on the datetime column
    df_combined = pd.merge(data_Grad, data_Temp, on="datetime", how="outer")

    # Convert datetime column to pandas datetime
    df_combined["datetime"] = pd.to_datetime(df_combined["datetime"])

    # Handle timezone alignment
    if df_combined["datetime"].dt.tz is None:  # If timezone-naive
        df_combined["datetime"] = df_combined["datetime"].dt.tz_localize("UTC").dt.tz_convert("Asia/Jerusalem")
    else:  # If already timezone-aware
        df_combined["datetime"] = df_combined["datetime"].dt.tz_convert("Asia/Jerusalem")

    # Round down to the nearest hour
    df_combined["hour"] = df_combined["datetime"].dt.ceil("h")

    # Group by the rounded hour and calculate the mean
    IMS_hourly = df_combined.groupby("hour").mean().reset_index()

    # Round Grad and TD to 1 decimal place
    IMS_hourly = IMS_hourly.round({"Grad": 1, "TD": 1})
    IMS_hourly = IMS_hourly.drop(columns=["datetime"], errors="ignore")

    # Ensure 'hour' column is timezone-aware
    IMS_hourly['hour'] = pd.to_datetime(IMS_hourly['hour']).dt.tz_convert("Asia/Jerusalem")

    # Set the 'hour' column as the index
    IMS_hourly.set_index('hour', inplace=True)

    print("IMS observed Hourly Averages:")
    print(IMS_hourly)
else:
    print("No data retrieved for one or both channels.")

# Open-Meteo API URL
api_url = "https://api.open-meteo.com/v1/forecast"

# Define request parameters
params = {
    "latitude": lat,
    "longitude": lon,
    "hourly": "temperature_2m,shortwave_radiation",  # Fetch temperature and GHI
    # "forecast_days": 1,  # Fetch data for 1 day
    "start_date": "2025-01-22",
    "end_date": "2025-01-22",
    "timezone": "Asia/Jerusalem"
}

# Make the API request
response = requests.get(api_url, params=params)

# Check for a valid response
if response.status_code == 200:
    # Parse the JSON response
    open_meteo_data = response.json()["hourly"]
    times = open_meteo_data["time"]
    temperatures = open_meteo_data["temperature_2m"]
    ghi_values = open_meteo_data["shortwave_radiation"]

    # Create a Pandas DataFrame
    open_meteo_df = pd.DataFrame({
        "datetime": pd.to_datetime(times),
        "Grad": ghi_values,
        "TD": temperatures
    })
    open_meteo_df["datetime"] = open_meteo_df["datetime"].dt.tz_localize("Asia/Jerusalem")
    print("Open-Meteo Temperature and GHI Data:")
    print(open_meteo_df)
else:
    print(f"Error fetching data from Open-Meteo API: {response.status_code}, {response.text}")



# Filter data for 22/1/2025
IMS_hourly_22 = IMS_hourly[IMS_hourly.index.date == pd.to_datetime('2025-01-22').date()]
open_meteo_df_22 = open_meteo_df[open_meteo_df["datetime"].dt.date == pd.to_datetime('2025-01-22').date()]

# Convert index to numerical representation for plotting (hours since epoch)
# IMS_hourly_22.index = IMS_hourly_22.index.astype(int) / 3.6e+12
# IMS_hourly_22.index = IMS_hourly_22.index


# open_meteo_df_22["datetime_num"] = open_meteo_df_22["datetime"].astype(int) / 3.6e+12

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Observed and Forecasted Temperature (Primary Y-axis)
ax1.plot(IMS_hourly_22.index + pd.Timedelta(hours=2), IMS_hourly_22["TD"], label="Observed Temperature", color="blue", linestyle="-")
# ax1.plot(open_meteo_df_22["datetime_num"], open_meteo_df_22["TD"], label="Forecasted Temperature", color="blue", linestyle="--")
ax1.plot(open_meteo_df_22["datetime"] + pd.Timedelta(hours=2), open_meteo_df_22["TD"], label="Forecasted Temperature", color="blue", linestyle="--")
ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Temperature (°C)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")

# Create a secondary Y-axis for Radiation
ax2 = ax1.twinx()

# Observed and Forecasted Radiation (Secondary Y-axis)
ax2.plot(IMS_hourly_22.index + pd.Timedelta(hours=2), IMS_hourly_22["Grad"], label="Observed Radiation", color="orange", linestyle="-")
# ax2.plot(open_meteo_df_22["datetime_num"], open_meteo_df_22["Grad"], label="Forecasted Radiation", color="orange", linestyle="--")
ax2.plot(open_meteo_df_22["datetime"] + pd.Timedelta(hours=2), open_meteo_df_22["Grad"], label="Forecasted Radiation", color="orange", linestyle="--")

ax2.set_ylabel("Radiation (W/m^2)", color="orange")
ax2.tick_params(axis='y', labelcolor="orange")


# display hour of day at x axis in UTC+02:


# Format x-axis to display hour of day
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H')) # Use %H for hour (0-23)

# Adjust x-axis limits if needed
# ax1.set_xlim([min(IMS_hourly_22.index.min(), open_meteo_df_22["datetime_num"].min()),
#              max(IMS_hourly_22.index.max(), open_meteo_df_22["datetime_num"].max())])


# Vertical lines - convert to numerical representation (hours since epoch)
ax1.axvline(x=pd.to_datetime('2025-01-22 13:00:00'), color='gray', linestyle='-.', linewidth=1.6)
# ax1.axvline(x=pd.to_datetime('2025-01-22 18:00:00'), color='gray', linestyle='-.', linewidth=1.6)
ax1.axvline(x=pd.to_datetime('2025-01-22 20:00:00'), color='green', linestyle='-.', linewidth=1.6)

# Title and Legend
plt.title("Observed and Forecasted Temperature and Radiation on 2025-01-22")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper left")

ax1.set_xlim(pd.to_datetime('2025-01-22 00:00'),
             pd.to_datetime('2025-01-22 23:00'))


# Grid
ax1.grid(True)

# Display plot
plt.show()



def simulate_tank_temperature(meteo, A_c, A_t, U_t, m, P_elec, start_time, stop_time, dt=3600, c_p=4186, eff=0.9, electric_heating_times=None, eff_elec=0.95, initial_temp=None, T_cap=100):
    """
    Simulate the evolution of the tank temperature over time.

    Parameters:
      meteo: Observed or forecasted hourly solar irradiance in W/m² and ambient temperatures in °C.
      electric_heating_times (array): Electric heating on off array.
      dt (float): Time step in seconds (e.g., 3600 for one hour).
      m (float): Mass of water in the tank (kg).
      c_p (float): Specific heat capacity of water (J/(kg·K)).
      A_c (float): Collector area (m²).
      eff (float): Combined solar collector efficiency (dimensionless).
      P_elec (float): Electric heater power in Watts.
      eff_elec (float): Efficiency of the electric heater.
      A_t (float): Effective surface area of the tank (m²).
      U_t (float): Overall heat loss coefficient of the tank (W/(m²·K)).
      initial_temp (float, optional): Initial tank temperature in °C.
                                      If not provided, uses the first value of ambient_temp.
      T_cap (float, optional): Maximum allowable tank temperature in °C (default is 100°C).
      start_time = first calculation hour (either sunrise or last observation, depending in the case)
      stop_time = last calculation hour (either last observation or shower time, depending in the case)

    Returns:
      T_tank (array): Array of tank temperatures (in °C) at each hourly step.
                      The array length is len(irradiation).
    """
    meteo["tank_temp"] = 0
    meteo["tank_temp"] = meteo["tank_temp"].astype(float)

    # set meteo["tank_temp"] at start_time to equal meteo["TD"] at start_time:
    meteo.loc[meteo.loc[start_time:stop_time].index[0], "tank_temp"] = float(meteo.loc[meteo.loc[start_time:stop_time].index[0], "TD"])


    # if initial_temp is None:
    #      meteo.loc[start_time:stop_time]["tank_temp"][0] = meteo.loc[start_time:stop_time]["TD"][0]
    # else:
    #     meteo.loc[start_time:stop_time]["tank_temp"][0] = initial_temp
    if initial_temp is None:
        meteo.loc[meteo.loc[start_time:stop_time].index[0], "tank_temp"] = meteo.loc[meteo.loc[start_time:stop_time].index[0]]["TD"]
    else:
        meteo.loc[meteo.loc[start_time:stop_time].index[0], "tank_temp"] = initial_temp
    for hour in meteo.loc[start_time:stop_time].index:
        irradiation = meteo.loc[hour, "Grad"]
        ambient_temp = meteo.loc[hour, "TD"]
        Q_solar = A_c * eff * irradiation * dt
        if electric_heating_times is not None:
            electric_heating_minutes = electric_heating_times.loc[hour, "hourly_heating"]
            Q_elec = P_elec * eff_elec * (electric_heating_minutes * 60)
        else:
            Q_elec = 0
        Q_loss = A_t * U_t * (meteo.loc[hour, "tank_temp"] - ambient_temp) * dt
        Q_net = Q_solar + Q_elec - Q_loss
        dT = Q_net / (m * c_p)
        meteo.loc[hour + timedelta(seconds=dt), "tank_temp"] = min(meteo.loc[hour, "tank_temp"] + dT, T_cap)
        # meteo.loc[hour, "tank_temp"] = min(meteo.loc[hour, "tank_temp"] + dT, T_cap)

    return meteo


# calculate temp from sunrise to now, based on observations:
start_time_sunrise = pd.to_datetime(sunrise)
stop_time_cut = pd.to_datetime('2025-01-22 13:00+02:00')
meteo = simulate_tank_temperature(IMS_hourly_22, A_c, A_t, U_t, m, P_elec, start_time_sunrise, stop_time_cut, dt=3600, c_p=4186, eff=0.9, electric_heating_times=None, eff_elec=0.95, initial_temp=None, T_cap=100)
print("Calculated tank temperature based on observed meteorologic parameters since sunrise:")
print(meteo.loc[start_time_sunrise:stop_time_cut])
last_observed_tank_temp = meteo.loc[meteo.loc[start_time_sunrise:stop_time_cut].index[-1]]["tank_temp"]
print("last observed temperature: ", last_observed_tank_temp)

# calculate temp from cut time to shower time, based on forecast:
start_time_cut = pd.to_datetime('2025-01-22 14:00+02:00')
stop_time_shower = pd.to_datetime('2025-01-22 20:00+02:00')
open_meteo_df_22["hour"] = open_meteo_df_22["datetime"]
open_meteo_df_22.set_index("hour", inplace=True)
meteo = simulate_tank_temperature(open_meteo_df_22, A_c, A_t, U_t, m, P_elec, start_time_cut, stop_time_shower, dt=3600, c_p=4186, eff=0.9, electric_heating_times=None, eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)
print("\nPredicted tank temperature based on forecasted meteorologic parameters from now on:")
print(meteo.loc[start_time_cut:stop_time_shower])
last_observed_tank_temp_showertime = meteo.loc[meteo.loc[start_time_cut:stop_time_shower].index[-1]]["tank_temp"]
print(f"predicted water temperature without heating: {last_observed_tank_temp_showertime:.2f} °C")


'''
# create fake pattern of electric heating activation with the index of IMS_hourly_22 and additional column "hourly heating" in jumps of 10 minutes:
hourly_heating_df = pd.DataFrame({
    "hourly_heating": np.random.choice([0, 10, 20, 30, 40, 50, 60], size=len(meteo))
}, index=meteo.index)

# Display the new DataFrame
print(hourly_heating_df)
# if min_tank_temp < last_observed_tank_temp_showertime:


simulated_heating = simulate_tank_temperature(open_meteo_df_22, A_c, A_t, U_t, m, P_elec, start_time_cut, stop_time_shower, dt=3600, c_p=4186, eff=0.9, electric_heating_times=hourly_heating_df, eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)
print("\nPredicted tank temperature based on forecasted meteorologic parameters from now on:")
print(simulated_heating.loc[start_time_cut:stop_time_shower])
last_observed_tank_temp_showertime_w_heating = simulated_heating.loc[simulated_heating.loc[start_time_cut:stop_time_shower].index[-1]]["tank_temp"]
print(f"predicted water temperature with heating: {last_observed_tank_temp_showertime_w_heating:.2f} °C")
'''

# create an algorithm to find the optimal heating pattern using MCTS (alternative to random hourly_heating_df):
if min_tank_temp > last_observed_tank_temp_showertime:
    print("The tank temperature is too low for the desired shower temperature.")
    print("Electric heating is required to reach the desired temperature.")
    print("The optimal heating pattern will be calculated using the Monte Carlo Tree Search (MCTS) algorithm.")

    # Define the MCTS algorithm to find the optimal heating pattern
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Heating pattern: list of heating minutes per hour
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0  # Represents the total time of heating (to be minimized)

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * np.sqrt(
            np.log(self.visits + 1) / (child.visits + 1e-6)))

    def expand(self, num_hours):
        new_state = self.state.copy()
        hour_to_modify = random.choice(range(num_hours))
        new_state[hour_to_modify] = random.choice([0, 10, 20, 30, 40, 50, 60])  # Possible heating times
        child_node = MCTSNode(new_state, parent=self)
        self.children.append(child_node)
        return child_node

    def update(self, reward):
        self.visits += 1
        self.value += reward


def mcts_optimization(meteo, start_time, stop_time, min_tank_temp, max_iterations=1000):
    num_hours = len(meteo.loc[start_time:stop_time])  # Get the correct number of hours
    root = MCTSNode(state=[0] * num_hours)  # Initialize state with the correct length

    optimization_history = []

    for iteration in range(max_iterations):
        node = root
        while node.is_fully_expanded():
            node = node.best_child()

        new_node = node.expand(num_hours)

        hourly_heating_df = pd.DataFrame({"hourly_heating": new_node.state},
                                         index=meteo.loc[start_time:stop_time].index)
        simulated_heating = simulate_tank_temperature(meteo, A_c, A_t, U_t, m, P_elec, start_time, stop_time, dt=3600,
                                                      c_p=4186, eff=0.9, electric_heating_times=hourly_heating_df,
                                                      eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)

        final_temp = simulated_heating.loc[stop_time]["tank_temp"]
        total_heating_time = sum(new_node.state)

        if final_temp >= min_tank_temp:
            reward = -total_heating_time  # Negative total heating time (to minimize it)
        else:
            reward = -1e6  # Large penalty if temperature condition is not met

        optimization_history.append((iteration, new_node.state, final_temp, total_heating_time, reward))

        while new_node is not None:
            new_node.update(reward)
            new_node = new_node.parent

    # Filter only successful iterations
    successful_iterations = [entry for entry in optimization_history if entry[2] >= min_tank_temp]

    # Select 10 equally spaced successful iterations from worst to best
    if successful_iterations:
        successful_iterations.sort(key=lambda x: x[4])  # Sort by reward
        selected_iterations = [successful_iterations[i] for i in
                               np.linspace(0, len(successful_iterations) - 1, 10, dtype=int)]
    else:
        selected_iterations = []

    # Include the worst iteration even if it didn't achieve the minimum temp
    worst_iteration = min(optimization_history, key=lambda x: x[4])
    selected_iterations.append(worst_iteration)

    # Select the best iteration from successful ones
    best_iteration = min(successful_iterations, key=lambda x: x[3]) if successful_iterations else worst_iteration
    best_schedule = best_iteration[1]

    visualize_optimization(selected_iterations, meteo, start_time, stop_time, min_tank_temp, best_schedule,
                           best_iteration)
    return best_schedule


def visualize_optimization(history, meteo, start_time, stop_time, min_tank_temp, best_schedule, best_iteration):
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    num_plots = len(history)
    colors = plt.cm.viridis(np.linspace(0, 1, num_plots))

    # Plot baseline case with no heating
    no_heating_df = pd.DataFrame({"hourly_heating": [0] * len(meteo.loc[start_time:stop_time])},
                                 index=meteo.loc[start_time:stop_time].index)
    no_heating_simulation = simulate_tank_temperature(meteo, A_c, A_t, U_t, m, P_elec, start_time, stop_time, dt=3600,
                                                      c_p=4186, eff=0.9, electric_heating_times=no_heating_df,
                                                      eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)
    axs[0].plot(no_heating_simulation.index, no_heating_simulation["tank_temp"], label='No Heating', linestyle='dotted',
                color='black')

    # Plot selected strategies with distinct colors
    for i, (iteration, schedule, temp, total_heating_time, reward) in enumerate(history):
        color = colors[i]
        hourly_heating_df = pd.DataFrame({"hourly_heating": schedule}, index=meteo.loc[start_time:stop_time].index)
        simulated_heating = simulate_tank_temperature(meteo, A_c, A_t, U_t, m, P_elec, start_time, stop_time, dt=3600,
                                                      c_p=4186, eff=0.9, electric_heating_times=hourly_heating_df,
                                                      eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)
        axs[0].plot(simulated_heating.index, simulated_heating["tank_temp"],
                    label=f'Iter {iteration}, Heat {total_heating_time} min, Reward {reward}', color=color, alpha=0.8)
        axs[1].plot(hourly_heating_df.index, hourly_heating_df["hourly_heating"], color=color, alpha=0.8)

    # Plot best strategy
    best_hourly_heating_df = pd.DataFrame({"hourly_heating": best_schedule},
                                          index=meteo.loc[start_time:stop_time].index)
    best_simulation = simulate_tank_temperature(meteo, A_c, A_t, U_t, m, P_elec, start_time, stop_time, dt=3600,
                                                c_p=4186, eff=0.9, electric_heating_times=best_hourly_heating_df,
                                                eff_elec=0.95, initial_temp=last_observed_tank_temp, T_cap=100)
    axs[0].plot(best_simulation.index, best_simulation["tank_temp"],
                label=f'Best Iter {best_iteration[0]}, Heat {best_iteration[3]} min, Reward {best_iteration[4]}',
                color='red', linewidth=2)
    axs[1].plot(best_hourly_heating_df.index, best_hourly_heating_df["hourly_heating"], color='red', linewidth=2)
    axs[0].axhline(y=min_tank_temp, color='red', linestyle='--', label='Min Required Temp')
    axs[0].set_xlim(pd.to_datetime(start_time), pd.to_datetime(stop_time))
    axs[1].set_xlim(pd.to_datetime(start_time), pd.to_datetime(stop_time))
    axs[0].set_ylabel("Tank Temperature (°C)")
    axs[1].set_ylabel("Heating Minutes Per Hour")
    axs[1].set_xlabel("Time")
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H'))  # Use %H for hour (0-23)

    plt.tight_layout()
    plt.show()


optimal_heating_schedule = mcts_optimization(open_meteo_df_22, start_time_cut, stop_time_shower, min_tank_temp, max_iterations=2000)

print("Optimized Heating Schedule (minutes per hour):")
for hour, minutes in zip(open_meteo_df_22.loc[start_time_cut:stop_time_shower].index, optimal_heating_schedule):
    print(f"{hour.strftime('%H:%M')}: {minutes} min")