import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config

#%%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():

    run_folder = Path("Best_HPC_ensemble_run")
    num_runs_ensemble = 5
    # iterate runs for each config file in the folder:
    for config_file in run_folder.glob("config_9.yml"):
        for num_runs_ensemble in range(num_runs_ensemble):
            run_config = Config(config_file)

            print('model:\t\t', run_config.model)
            print('seq_length:\t', run_config.seq_length)
            print('dynamic_inputs:', run_config.dynamic_inputs)
            if torch.cuda.is_available():
                start_run(config_file=config_file)

                # fall back to CPU-only mode
            else:
                start_run(config_file=config_file,
                          gpu=-1)


if __name__ == '__main__':
    main()
