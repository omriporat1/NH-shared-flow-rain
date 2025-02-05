import inspect
from typing import Dict
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
from pathlib import Path
from neuralhydrology.evaluation import metrics

from neuralhydrology.modelzoo import get_model
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config
from neuralhydrology.nh_run import start_run, eval_run



def main():
    if torch.cuda.is_available():
        start_run(config_file=Path("1_basin_T2.yml"))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("1_basin_T2.yml"), gpu=-1)


    run_dir = Path(r"S:\hydrolab\home\Omri_Porat\PhD\Python\neuralhydrology-neuralhydrology-e4329c3\neuralhydrology\T1\runs\test_run2_1908_172449")
    eval_run(run_dir=run_dir, period="test")


    with open(run_dir / "test" / "model_epoch030" / "test_results.p", "rb") as fp:
        results = pickle.load(fp)

    results.keys()
    print(results['01022500']['1D']['xr'])

    a = 1

    # extract observations and simulations
    qobs = results['01022500']['1D']['xr']['QObs(mm/d)_obs']
    qsim = results['01022500']['1D']['xr']['QObs(mm/d)_sim']

    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(qobs['date'], qobs)
    ax.plot(qsim['date'], qsim)
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - NSE {results['01022500']['1D']['NSE']:.3f}")
    plt.show()

    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    for key, val in values.items():
        print(f"{key}: {val:.3f}")

    qobs = results['01013500']['1D']['xr']['QObs(mm/d)_obs']
    qsim = results['01013500']['1D']['xr']['QObs(mm/d)_sim']

    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(qobs['date'], qobs)
    ax.plot(qsim['date'], qsim)
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - NSE {results['01013500']['1D']['NSE']:.3f}")
    plt.show()

    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    for key, val in values.items():
        print(f"{key}: {val:.3f}")

    qobs = results['01123000']['1D']['xr']['QObs(mm/d)_obs']
    qsim = results['01123000']['1D']['xr']['QObs(mm/d)_sim']

    fig, ax = plt.subplots(figsize=(16,10))
    ax.plot(qobs['date'], qobs)
    ax.plot(qsim['date'], qsim)
    ax.set_ylabel("Discharge (mm/d)")
    ax.set_title(f"Test period - NSE {results['01123000']['1D']['NSE']:.3f}")
    plt.show()

    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    for key, val in values.items():
        print(f"{key}: {val:.3f}")

    a = 1


if __name__ == '__main__':
    main()






# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if __name__ == '__main__':
    if torch.cuda.is_available():
        start_run(config_file=Path("1_basin_T2.yml"))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("1_basin_T2.yml"), gpu=-1)


