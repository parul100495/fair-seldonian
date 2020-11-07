#!/bin/bash
#SBATCH --job-name=opt-small%j
#SBATCH -p titanx-long
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=logs-opt-%j.out
#SBATCH --mem=50000
module load python3/current

srun python3 main.py opt


