#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --job-name=flControl
#SBATCH --output=flControl.out
#SBATCH --error=flControl.err

cd ./../L
python training.py --name fl_control --params configs/control.yaml --commit none
