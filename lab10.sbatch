#!/bin/bash
#SBATCH --job-name=lab10
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/deep_learning/env_dl/bin/activate"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export MPLBACKEND=Agg

hostname
date
srun python -u lab10.py
date
