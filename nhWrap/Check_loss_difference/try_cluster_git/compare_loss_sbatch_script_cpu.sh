#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

source /usr/local/spack/opt/spack/linux-debian12-x86_64/gcc-12.2.0/miniconda3-24.3.0-iqeknetqo7ngpr57d6gmu3dg4rzlcgk6/etc/profile.d/conda.sh

conda activate /sci/labs/efratmorin/haimasree/condaenvs/neuralhydrology

export PYTHONPATH=/sci/labs/efratmorin/omripo/PhD/NH-shared-flow-rain/nhWrap/neuralhydrology:$PYTHONPATH

echo "PYTHONPATH is: $PYTHONPATH"

python /sci/labs/efratmorin/omripo/PhD/NH-shared-flow-rain/nhWrap/Check_loss_difference/try_cluster_git/check_loss_difference.py

conda deactivate

