#!/bin/bash
#SBATCH --job-name=plot-%j
#SBATCH -p titanx-short
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-plot%j.out
#SBATCH --mem=5000
module load cuda90/toolkit
module load python3/current

srun python3 ./create_plots.py
