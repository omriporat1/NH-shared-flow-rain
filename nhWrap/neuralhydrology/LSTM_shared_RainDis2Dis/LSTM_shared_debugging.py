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
    run_config = Config(Path("LSTM_shared_config_NoBadData.yml"))

    print('model:\t\t', run_config.model)
    print('seq_length:\t', run_config.seq_length)
    print('dynamic_inputs:', run_config.dynamic_inputs)
    if torch.cuda.is_available():
        start_run(config_file=Path(
            "LSTM_shared_config_NoBadData.yml"))

        # fall back to CPU-only mode
    else:
        start_run(config_file=Path(
            "LSTM_shared_config_NoBadData.yml"),
                  gpu=-1)


if __name__ == '__main__':
    main()
