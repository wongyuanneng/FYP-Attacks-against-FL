#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=8000M
#SBATCH --job-name=pdganTest
#SBATCH --output=pdganTest.out
#SBATCH --error=pdganTest.err

cd ./../
python training.py --name pdgan_test --params configs/pdgan_test.yaml --commit none
