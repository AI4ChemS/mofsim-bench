#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=5000
#SBATCH --gres=gpu:1
#SBATCH --job-name=mb_stability
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.out
#SBATCH --array 0-3

CALCULATOR=$1
SETTINGS=$2

# Read the calculator name, must be in the format "calculator_suffix"
IFS="_"
read -ra parts <<< "$CALCULATOR"
unset IFS

conda activate "mb_${parts[0]}"

srun -u python -u stability.py --calculator $CALCULATOR --settings $SETTINGS --index $SLURM_ARRAY_TASK_ID