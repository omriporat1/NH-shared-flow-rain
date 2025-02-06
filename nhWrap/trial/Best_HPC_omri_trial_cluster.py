import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from nhWrap.neuralhydrology.evaluation import metrics, get_tester
from nhWrap.neuralhydrology.nh_run import start_run
from nhWrap.neuralhydrology.utils.config import Config

#%%
# by default we assume that you have at least one CUDA-capable NVIDIA GPU


def main():
    print("running correct file")
    if torch.cuda.is_available():
        start_run(config_file=Path(r"\sci\labs\efratmorin\omripo\PhD\NH-shared-flow-rain\nhWrap\trial\config_omri_gpu.yml"))


if __name__ == '__main__':
    main()
