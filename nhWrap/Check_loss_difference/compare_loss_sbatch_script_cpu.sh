#!/bin/bash

#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

cd sci/labs/efratmorin/omripo/PhD/NH-shared-flow-rain/nhWrap/Check_loss_difference

. /sci/labs/efratmorin/omripo/PhD/Python/neuralhydrology-neuralhydrology-e4329c3/NH_env/neuralhydrology/bin/activate

python /sci/labs/efratmorin/omripo/PhD/NH-shared-flow-rain/nhWrap/Check_loss_difference/check_loss_difference.py

deactivate

