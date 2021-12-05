#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --job-name=pdganControl
#SBATCH --output=pdganControl.out
#SBATCH --error=pdganControl.err

cd ./../
python training.py --name pdgan_control --params configs/pdgan_control.yaml --commit none
