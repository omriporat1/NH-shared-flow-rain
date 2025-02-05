import csv
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
import os
import xarray
from datetime import datetime



# %%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU

def log_metrics_to_csv(csv_path, model_name, basin, metric_results, model_params):
    """
    Logs model metrics into a CSV file. Appends new rows for each model-basin combination.

    Args:
        csv_path (str): Path to the CSV file.
        model_name (str): Name of the trained model.
        basin (str): Basin identifier.
        metric_results (dict): Dictionary of metric results.
    """
    csv_file = Path(csv_path)
    # Check if file exists to determine whether to write headers
    file_exists = csv_file.exists()

    # Prepare data for logging
    row = {
        "model_name": model_name,
        "basin": basin,
        **model_params,  # Include model parameters
        **{key: f"{value:.3g}" for key, value in metric_results.items()}  # Log metrics with 3 significant digits
    }

    # Write to the CSV file
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()  # Write header if file is new
        writer.writerow(row)

def extract_model_params(config):
    """
    Extracts model parameters from the configuration file.

    Args:
        config (Config): NeuralHydrology Config object.

    Returns:
        dict: Dictionary of model parameters.
    """
    params_to_extract = ["learning_rate", "seq_length", "hidden_size", "batch_size", "epochs", "autoregressive_inputs", "dynamic_inputs"]  # Adjust as needed
    return {param: getattr(config, param, None) for param in params_to_extract}


def main():
    figures_dir = Path("Best_HPC_ensemble_run")
    runs_dir = Path("runs")

    max_events_path = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3"
                       r"\neuralhydrology\max_event_dates.csv")
    max_event_per_basin = pd.read_csv(max_events_path)
    max_event_per_basin = max_event_per_basin.set_index("basin")

    patterns = [
        "HPC_training_zscore_norm_hidden_size256_batch_size512_learning_rate0001_2812_002447",
        "Best_HPC_training_zscore_norm_ensemble_hidden_size256_batch_size512_learning_rate0001_2812_233440",
        "Best_HPC_training_zscore_norm_ensemble_hidden_size256_batch_size512_learning_rate0001_2912_033901",
        "Best_HPC_training_zscore_norm_ensemble_hidden_size256_batch_size512_learning_rate0001_2912_101027",
        "Best_HPC_training_zscore_norm_ensemble_hidden_size256_batch_size512_learning_rate0001_2912_164318"
    ]
    model_dirs = [path for pattern in patterns for path in runs_dir.glob(pattern)]
    # csv_path = "run_metrics_by_basin.csv"

    '''
    qobs = None
    qsim = None
    qobs_dates = None
    qsim_dates = None
    qobs_shift_dates = None
    values = None
    results = None
    '''

    qobs = {}
    qsim = {}
    qobs_dates = {}
    qsim_dates = {}
    qobs_shift = {}
    qobs_shift_dates = {}
    values = {}
    ref_values = {}

    model_params = {}
    basins = {}

    for i, run_dir in enumerate(model_dirs):
        model_name = run_dir.name
        print(f"Evaluating model: {model_name}")
        run_config = Config(run_dir / "config.yml")
        model_params[i] = extract_model_params(run_config)

        data_dir = run_config.data_dir

        max_events_path = (r"C:\PhD\Python\neuralhydrology-neuralhydrology-e4329c3"
                           r"\neuralhydrology\max_event_dates.csv")
        # create a tester instance and start evaluation
        tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
        results = tester.evaluate(save_results=False, metrics=run_config.metrics)

        # results.keys()
        basins[i] = results.keys()


        # Normalization method:
        # normalization = 'minmax_norm'
        normalization = 'z_score_norm'
        # normalization = 'unit_discharge'
        # normalization = 'no_norm'

        norm_dict = pickle.load(open(data_dir / r"timeseries\netcdf\il\normalization_dict.pkl", "rb"))

        max_event_per_basin = pd.read_csv(max_events_path)
        max_event_per_basin = max_event_per_basin.set_index("basin")

        delay = 18  # 3 hours = 18 10-minute steps

        for basin in basins[i]:

            start_date = max_event_per_basin.loc[basin, "event_start"]
            end_date = max_event_per_basin.loc[basin, "event_end"]
            start_date = datetime.strptime(start_date, "%d/%m/%Y")
            end_date = datetime.strptime(end_date, "%d/%m/%Y")

            if basin not in qobs:
                qobs[basin] = {}
                qsim[basin] = {}
                qobs_dates[basin] = {}
                qsim_dates[basin] = {}
                qobs_shift_dates[basin] = {}
                values[basin] = {}

            start_date = max_event_per_basin.loc[basin, "event_start"]
            end_date = max_event_per_basin.loc[basin, "event_end"]

            # use relevant normalization values of the basin to rescale the data:
            basin_norm_dict = norm_dict[basin]
            if normalization == 'minmax_norm':
                # extract observations and simulations for a specific station in a specific date range:
                qobs[basin][i] = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_obs"]
                qsim[basin][i] = results[basin]["10min"]["xr"]["Flow_m3_sec_minmax_norm_sim"]
                min_values = basin_norm_dict["features"]["Flow_m3_sec"]["min"]
                max_values = basin_norm_dict["features"]["Flow_m3_sec"]["max"]
                qobs[basin][i] = qobs[basin][i] * (max_values - min_values) + min_values
                qsim[basin][i] = qsim[basin][i] * (max_values - min_values) + min_values
                norm_type = 'training minmax normalization'

            elif normalization == 'z_score_norm':
                # extract observations and simulations for a specific station in a specific date range:
                qobs[basin][i] = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"]
                qsim[basin][i] = results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"]
                mean_values = basin_norm_dict["features"]["Flow_m3_sec"]["mean"]
                std_values = basin_norm_dict["features"]["Flow_m3_sec"]["std"]
                qobs[basin][i] = qobs[basin][i] * std_values + mean_values
                qsim[basin][i] = qsim[basin][i] * std_values + mean_values
                norm_type = 'training z-score normalization'

            elif normalization == 'unit_discharge':
                # extract observations and simulations for a specific station in a specific date range:
                qobs[basin][i] = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_obs"]
                qsim[basin][i] = results[basin]["10min"]["xr"]["unit_discharge_m3_sec_km_sim"]
                basin_area = basin_norm_dict["basin_area"]
                qobs[basin][i] = qobs[basin][i] * basin_area
                qsim[basin][i] = qsim[basin][i] * basin_area
                norm_type = 'unit discharge normalization'

            else:
                # extract observations and simulations for a specific station in a specific date range:
                qobs[basin][i] = results[basin][i]["10min"]["xr"]["Flow_m3_sec_obs"]
                qsim[basin][i] = results[basin][i]["10min"]["xr"]["Flow_m3_sec_sim"]
                norm_type = 'no normalization'

            # Shift the observed data by the delay
            fill_value = qobs[basin][i].isel(date=0, time_step=0).item()
            qobs_shift[basin] = qobs[basin][i].shift(date=18, fill_value=fill_value)

            # Filter qobs based on the date range
            qobs_dates[basin][i] = qobs[basin][i].sel(date=slice(start_date, end_date))
            qsim_dates[basin][i] = qsim[basin][i].sel(date=slice(start_date, end_date))

            values[basin][i] = metrics.calculate_all_metrics(qobs[basin][i].isel(time_step=-1), qsim[basin][i].isel(time_step=-1), "10min")

            # c
            if i == 0:
                ref_values[basin] = metrics.calculate_all_metrics(qobs[basin][i].isel(time_step=-1), qobs_shift[basin].isel(time_step=-1), "10min")
                qobs_shift_dates[basin] = qobs_shift[basin].sel(date=slice(start_date, end_date))


    # for each basin, calculate a median qsim:
    qsim_median = {}
    ensemble_metrics = {}

    for basin in basins[0]:
        start_date = max_event_per_basin.loc[basin, "event_start"]
        end_date = max_event_per_basin.loc[basin, "event_end"]
        start_date = datetime.strptime(start_date, "%d/%m/%Y")
        end_date = datetime.strptime(end_date, "%d/%m/%Y")

        qsim_median[basin] = xarray.concat([qsim[basin][i] for i in range(len(model_dirs))], dim='model').median(dim='model')

        # calculate metrics for the ensemble:
        ensemble_metrics[basin] = metrics.calculate_all_metrics(qobs[basin][0].isel(time_step=-1), qsim_median[basin].isel(time_step=-1), "10min")

        # create a figure for the basin with the observed in blue, the shifted observed in green, the simulated members in grey and the ensemble median in black:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(qobs[basin][0]["date"], qobs[basin][0], label="Observed", color='blue')
        ax.plot(qobs_shift[basin]["date"], qobs_shift[basin], label="Observed shifted", color='green')
        for i in range(len(model_dirs)):
            ax.plot(qsim[basin][i]["date"], qsim[basin][i], label=f"Simulated {i}", color='grey', alpha=0.5)
        ax.plot(qsim_median[basin]["date"], qsim_median[basin], label="Ensemble median", color='orange')
        ax.grid()
        ax.set_ylabel("Discharge (m^3/s)")
        ax.set_xlim(datetime(2018, 1, 1), datetime(2018, 3, 1))
        ax.set_title(f"Test period - basin {basin} - ensemble median")
        ax.legend()

        # add the ensemble metrics to the ensemble_metrics figure:
        ax.text(0.01, 0.96, "Ensemble metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.01, 0.92, f"NSE: {ensemble_metrics[basin]['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.01, 0.88, f"RMSE: {ensemble_metrics[basin]['RMSE']:.2f}", transform=ax.transAxes)
        # ax.text(0.01, 0.84, f"FHV: {ensemble_metrics[basin]['FHV']:.1f}", transform=ax.transAxes)

        # add the reference metrics to the ensemble_metrics figure:
        ax.text(0.15, 0.96, "Reference metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.15, 0.92, f"NSE: {ref_values[basin]['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.15, 0.88, f"RMSE: {ref_values[basin]['RMSE']:.2f}", transform=ax.transAxes)
        # ax.text(0.23, 0.84, f"FHV: {ref_values[basin]['FHV']:.1f}", transform=ax.transAxes)

        plt.show()

        # Save the figure
        fig.savefig(figures_dir / f"{basin}_ensemble.png")

        # create a figure for the basin with the observed in blue, the shifted observed in green, the simulated members in grey and the ensemble median in black:
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.plot(qobs[basin][0]["date"], qobs[basin][0], label="Observed", color='blue')
        ax.plot(qobs_shift[basin]["date"], qobs_shift[basin], label="Observed shifted", color='green')
        for i in range(len(model_dirs)):
            ax.plot(qsim[basin][i]["date"], qsim[basin][i], label=f"Simulated {i}", color='grey', alpha=0.5)
        ax.plot(qsim_median[basin]["date"], qsim_median[basin], label="Ensemble median", color='orange')
        ax.grid()
        ax.set_ylabel("Discharge (m^3/s)")
        ax.set_title(f"Test period - basin {basin} - ensemble median - zoom")
        ax.legend()

        # add the ensemble metrics to the ensemble_metrics figure:
        ax.text(0.01, 0.96, "Ensemble metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.01, 0.92, f"NSE: {ensemble_metrics[basin]['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.01, 0.88, f"RMSE: {ensemble_metrics[basin]['RMSE']:.2f}", transform=ax.transAxes)
        # ax.text(0.01, 0.84, f"FHV: {ensemble_metrics[basin]['FHV']:.1f}", transform=ax.transAxes)

        # add the reference metrics to the ensemble_metrics figure:
        ax.text(0.15, 0.96, "Reference metrics", transform=ax.transAxes, fontweight='bold')
        ax.text(0.15, 0.92, f"NSE: {ref_values[basin]['NSE']:.2f}", transform=ax.transAxes)
        ax.text(0.15, 0.88, f"RMSE: {ref_values[basin]['RMSE']:.2f}", transform=ax.transAxes)
        # ax.text(0.23, 0.84, f"FHV: {ref_values[basin]['FHV']:.1f}", transform=ax.transAxes)

        ax.set_xlim(start_date, end_date)

        plt.show()
        fig.savefig(figures_dir / f"{basin}_ensemble_zoom.png")

    # save the ref_values and ensemble_metrics to a csv file:
    ref_values_df = pd.DataFrame(ref_values).T
    ensemble_metrics_df = pd.DataFrame(ensemble_metrics).T
    ref_values_df.to_csv(figures_dir / "ref_values.csv")
    ensemble_metrics_df.to_csv(figures_dir / "ensemble_metrics.csv")


if __name__ == '__main__':
    main()
