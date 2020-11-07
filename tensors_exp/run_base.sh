#!/bin/bash
#SBATCH --job-name=base-small%j
#SBATCH -p 2080ti-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-base-%j.out
#SBATCH --mem=50000
module load python3/current

srun python3 main.py base


