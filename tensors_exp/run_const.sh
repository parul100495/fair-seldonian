#!/bin/bash
#SBATCH --job-name=const-small%j
#SBATCH -p m40-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-const-%j.out
#SBATCH --mem=50000
module load python3/current

srun python3 main.py const


