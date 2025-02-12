import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import eval_run
from neuralhydrology.utils.config import Config
import os
import xarray as xr
from datetime import datetime


# %%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    run_dir = Path(
        r"C:\PhD\Python\NH-shared-flow-rain\nhWrap\neuralhydrology\LSTM_shared_RainDis2Dis\runs"
        r"\Best_HPC_training_zscore_norm_hidden_size256_batch_size512_learning_rate0001_1002_162559")  #
    # you'll find this path in the output of the training above.
    run_config = Config(Path(
        r"C:\PhD\Python\NH-shared-flow-rain\nhWrap\neuralhydrology\LSTM_shared_RainDis2Dis"
        r"\Apply_metrics_to_train_and_validation\Best_HPC_config.yml"))

    num_epochs = run_config.epochs
    # initialize the metrics storage:
    metric_per_epoch_train = pd.DataFrame(
        columns=["NSE", "MSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "FHV", "FMS", "FLV"])
    metric_per_epoch_val = pd.DataFrame(
        columns=["NSE", "MSE", "KGE", "Alpha-NSE", "Beta-NSE", "Pearson-r", "FHV", "FMS", "FLV"])

    # iterate runs for each config file in the folder:
    for epoch in range(num_epochs):
        eval_run(run_dir=run_dir, period="train", epoch=epoch+1, gpu=-1)
        eval_run(run_dir=run_dir, period="validation", epoch=epoch+1, gpu=-1)

        with open(run_dir / "train" / f"model_epoch{epoch+1:03d}" / "train_results.p", "rb") as fp:
            train_results = pickle.load(fp)
        with open(run_dir / "validation" / f"model_epoch{epoch+1:03d}" / "validation_results.p", "rb") as fp:
            val_results = pickle.load(fp)

        '''
        # extract observations and simulations for each of the basins:
        qobs_10_min_train = []
        qsim_10_min_train = []

        qobs_10_min_val = []
        qsim_10_min_val = []
        '''

        all_results_train_sim = [train_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"] for basin in train_results.keys()]
        qsim_10_min_train = xr.concat(all_results_train_sim, dim='date')
        all_results_train_obs = [train_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"] for basin in train_results.keys()]
        qobs_10_min_train = xr.concat(all_results_train_obs, dim='date')

        all_results_val_sim = [val_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"] for basin in val_results.keys()]
        qsim_10_min_val = xr.concat(all_results_val_sim, dim='date')
        all_results_val_obs = [val_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"] for basin in val_results.keys()]
        qobs_10_min_val = xr.concat(all_results_val_obs, dim='date')

        '''
        for basin in train_results.keys():
            qsim_10_min_train = qsim_10_min_train.append(train_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"])
            qobs_10_min_train = qobs_10_min_train.append(train_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"])
        '''
        values_train = metrics.calculate_all_metrics(qobs_10_min_train.isel(time_step=-1), qsim_10_min_train.isel(time_step=-1))

        '''
        for basin in val_results.keys():
            qsim_10_min_val = qsim_10_min_val.append(val_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_sim"])
            qobs_10_min_val = qobs_10_min_val.append(val_results[basin]["10min"]["xr"]["Flow_m3_sec_zscore_norm_obs"])
        '''
        values_val = metrics.calculate_all_metrics(qobs_10_min_val.isel(time_step=-1), qsim_10_min_val.isel(time_step=-1))

        # calculate some metrics for the training and validation period of the current epoch:
        '''
        print(f"Epoch: {epoch}")
        print("10-min metrics:")
        for key, val in values_train.items():
            print(f"train  {key}: {val:.3f}")
        for key, val in values_val.items():
            print(f"val  {key}: {val:.3f}")
        '''
        # save metrics in the dataframe:
        metric_per_epoch_train.loc[epoch, "NSE"] = values_train["NSE"]
        metric_per_epoch_train.loc[epoch, "MSE"] = values_train["MSE"]
        metric_per_epoch_train.loc[epoch, "KGE"] = values_train["KGE"]
        metric_per_epoch_train.loc[epoch, "Alpha-NSE"] = values_train["Alpha-NSE"]
        metric_per_epoch_train.loc[epoch, "Beta-NSE"] = values_train["Beta-NSE"]
        metric_per_epoch_train.loc[epoch, "Pearson-r"] = values_train["Pearson-r"]
        metric_per_epoch_train.loc[epoch, "FHV"] = values_train["FHV"]
        metric_per_epoch_train.loc[epoch, "FMS"] = values_train["FMS"]
        metric_per_epoch_train.loc[epoch, "FLV"] = values_train["FLV"]

        metric_per_epoch_val.loc[epoch, "NSE"] = values_val["NSE"]
        metric_per_epoch_val.loc[epoch, "MSE"] = values_val["MSE"]
        metric_per_epoch_val.loc[epoch, "KGE"] = values_val["KGE"]
        metric_per_epoch_val.loc[epoch, "Alpha-NSE"] = values_val["Alpha-NSE"]
        metric_per_epoch_val.loc[epoch, "Beta-NSE"] = values_val["Beta-NSE"]
        metric_per_epoch_val.loc[epoch, "Pearson-r"] = values_val["Pearson-r"]
        metric_per_epoch_val.loc[epoch, "FHV"] = values_val["FHV"]
        metric_per_epoch_val.loc[epoch, "FMS"] = values_val["FMS"]
        metric_per_epoch_val.loc[epoch, "FLV"] = values_val["FLV"]


    # Create a plot with subplot for each metric, showing the training and validation performance over the epochs:
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i, metric in enumerate(metric_per_epoch_train.columns):
        ax = axs.flatten()[i]
        ax.plot(metric_per_epoch_train[metric], label="train")
        ax.plot(metric_per_epoch_val[metric], label="val")
        ax.set_title(metric)
        ax.legend()
    plt.tight_layout()
    plt.show()

    # save the metrics to a csv file:
    metric_per_epoch_train.to_csv(run_dir / "train_metrics.csv")
    metric_per_epoch_val.to_csv(run_dir / "val_metrics.csv")


if __name__ == '__main__':
    main()
