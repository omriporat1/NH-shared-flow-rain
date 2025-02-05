import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config


# %%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    run_config = Config(Path("LSTM_shared_config_check_delay.yml"))
    run_dir = Path(
        "runs/LSTM_shared_debugging_runs_check_delay_1_2411_155931")  # you'll find this path in the output of the training above.

    # create a tester instance and start evaluation
    tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
    results = tester.evaluate(save_results=False, metrics=run_config.metrics)

    results.keys()

    # extract observations and simulations
    qobs = results["il_8146"]["10min"]["xr"]["Flow_m3_sec_obs"]
    qsim = results["il_8146"]["10min"]["xr"]["Flow_m3_sec_sim"]

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(qobs["date"], qobs, label="Observed")
    ax.plot(qsim["date"], qsim, label="Simulated")
    ax.legend()
    ax.set_ylabel("Discharge (mm/h)")
    ax.set_title(f"Test period - daily NSE {results['il_8146']['10min']['NSE']:.3f}")
    plt.show()

    # Calculate some metrics
    values = metrics.calculate_all_metrics(qobs.isel(time_step=-1), qsim.isel(time_step=-1))
    print("10-minute metrics:")
    for key, val in values.items():
        print(f"  {key}: {val:.3f}")


if __name__ == '__main__':
    main()
