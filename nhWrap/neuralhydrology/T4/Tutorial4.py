import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config

#%%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU

'''
def main():
    if torch.cuda.is_available():
        start_run(config_file=Path("S:/hydrolab/home/Omri_Porat/PhD/Python/neuralhydrology-neuralhydrology-e4329c3/neuralhydrology/T1/1_basin.yml"))

    # fall back to CPU-only mode
    else:
        start_run(config_file=Path("S:/hydrolab/home/Omri_Porat/PhD/Python/neuralhydrology-neuralhydrology-e4329c3/neuralhydrology/T1/1_basin.yml"), gpu=-1)


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
'''
def main():
    run_config = Config(Path("1_basin.yml"))
    print('model:\t\t', run_config.model)
    print('use_frequencies:', run_config.use_frequencies)
    print('seq_length:\t', run_config.seq_length)
    print('dynamic_inputs:', run_config.dynamic_inputs)
    if torch.cuda.is_available():
        start_run(config_file=Path(
            "S:/hydrolab/home/Omri_Porat/PhD/Python/neuralhydrology-neuralhydrology-e4329c3/neuralhydrology/T4/1_basin.yml"))

        # fall back to CPU-only mode
    else:
        start_run(config_file=Path(
            "S:/hydrolab/home/Omri_Porat/PhD/Python/neuralhydrology-neuralhydrology-e4329c3/neuralhydrology/T4/1_basin.yml"),
                  gpu=-1)

    a=1

    run_dir = Path("runs/test_run_2208_105722")  # you'll find this path in the output of the training above.

    # create a tester instance and start evaluation
    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=False, metrics=run_config.metrics)

    results.keys()

    # extract observations and simulations
    daily_qobs = results["01022500"]["1D"]["xr"]["qobs_mm_per_hour_obs"]
    daily_qsim = results["01022500"]["1D"]["xr"]["qobs_mm_per_hour_sim"]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(daily_qobs["date"], daily_qobs, label="Observed")
    ax.plot(daily_qsim["date"], daily_qsim, label="Simulated")
    ax.legend()
    ax.set_ylabel("Discharge (mm/h)")
    ax.set_title(f"Test period - daily NSE {results['01022500']['1D']['NSE_1D']:.3f}")

    # Calculate some metrics
    values = metrics.calculate_all_metrics(daily_qobs.isel(time_step=-1), daily_qsim.isel(time_step=-1))
    print("Daily metrics:")
    for key, val in values.items():
        print(f"  {key}: {val:.3f}")

    # extract a date slice of observations and simulations
    hourly_xr = results["01022500"]["1h"]["xr"].sel(date=slice("10-1995", None))

    # The hourly data is indexed with two indices: The date (in days) and the time_step (the hour within that day).
    # As we want to get a continuous plot of several days' hours, we select all 24 hours of each day and then stack
    # the two dimensions into one consecutive datetime dimension.
    hourly_xr = hourly_xr.isel(time_step=slice(-24, None)).stack(datetime=['date', 'time_step'])
    hourly_xr['datetime1'] = hourly_xr.coords['date'] + hourly_xr.coords['time_step']

    hourly_qobs = hourly_xr["qobs_mm_per_hour_obs"]
    hourly_qsim = hourly_xr["qobs_mm_per_hour_sim"]
    hourly_qobs["datetime"] = pd.to_datetime(hourly_qobs["datetime"])

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(hourly_qobs["date"], hourly_qobs, label="Observation")
    ax.plot(hourly_qsim["date"], hourly_qsim, label="Simulation")
    ax.set_ylabel("Discharge (mm/h)")
    ax.set_title(f"Test period - hourly NSE {results['01022500']['1h']['NSE_1h']:.3f}")
    _ = ax.legend()

    plt.show()
    a=1

if __name__ == '__main__':
    main()
