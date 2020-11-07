#!/bin/bash
#SBATCH --job-name=random-main-%j
#SBATCH -p titanx-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-random-%j.out
#SBATCH --mem=50000
module load cuda90/toolkit
module load python3/current

srun python3 ./main_plotting.py
