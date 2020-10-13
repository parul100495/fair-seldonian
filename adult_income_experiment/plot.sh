#!/bin/bash
#SBATCH --job-name=adult-main-%j
#SBATCH -p m40-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-create%j.out
#SBATCH --mem=5000
module load cuda90/toolkit
module load python3/current

srun python3 ./create_plots_adult.py
