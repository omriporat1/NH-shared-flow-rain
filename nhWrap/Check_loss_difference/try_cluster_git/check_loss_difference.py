#!/usr/bin/env python
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from neuralhydrology.evaluation import metrics, get_tester
from neuralhydrology.nh_run import start_run
from neuralhydrology.utils.config import Config
# from nhWrap.neuralhydrology.evaluation import metrics, get_tester
# from nhWrap.neuralhydrology.nh_run import start_run
# from nhWrap.neuralhydrology.utils.config import Config

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

#%%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():

    run_folder = Path(".")
    num_runs_ensemble = 1
    # iterate runs for each config file in the folder:
    for config_file in run_folder.glob("*.yml"):
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
