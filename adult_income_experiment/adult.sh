#!/bin/bash
#SBATCH --job-name=adult-main-%j
#SBATCH -p titanx-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs%j.out
#SBATCH --mem=5000
module load cuda90/toolkit
module load python3/current

srun python3 ./main_adult.py
