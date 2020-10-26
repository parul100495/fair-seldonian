#!/bin/sh
#SBATCH -J hoeffding
#SBATCH --mem=50000 --partition=longq --nodes=1 --ntasks=1 --cpus-per-task=1
#SBATCH --output=logs-hoeffding-create%j.out

export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export OMP_NUM_THREADS=24

srun python3 create_plots.py


