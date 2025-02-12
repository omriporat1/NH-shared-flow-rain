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
    print("running correct file")
    # if torch.cuda.is_available():
    #     print("cuda is available")
    start_run(config_file=Path(r"C:\PhD\Python\NH-shared-flow-rain\nhWrap\neuralhydrology\LSTM_shared_RainDis2Dis\working_configuration_for_run_Best_HPC_training_zscore_norm_ensemble_hidden_size256_batch_size512_learning_rate0001_1002_235205\config_omri_gpu.yml"), gpu=-1)


if __name__ == '__main__':
    main()
