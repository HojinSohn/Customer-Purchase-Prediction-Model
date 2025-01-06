#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --job-name XGBtest
#SBATCH --output XGBtest.out
#SBATCH --error XGBtest.err

# Run python file.

# Load our conda environment
module load anaconda/2024.02-py311
source activate CS373

# Run the test code
python3 ~/KaggleCompetition/test.py
