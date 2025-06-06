#!/bin/bash -l

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:5:00
#SBATCH --mem=5000
#SBATCH --gres=gpu:1
#SBATCH --job-name=mb_ie
#SBATCH --partition=gpu_h100,gpu_a100_il,gpu_h100_il #,dev_gpu_h100
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.out

CALCULATOR=$1
SETTINGS=$2

# Read the calculator name, must be in the format "calculator_suffix"
IFS="_"
read -ra parts <<< "$CALCULATOR"
unset IFS

conda activate "mb_${parts[0]}"

srun -u python interaction_energy.py --calculator $CALCULATOR --settings $SETTINGS