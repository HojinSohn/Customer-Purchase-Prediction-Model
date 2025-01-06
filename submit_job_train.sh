#!/bin/bash
#SBATCH -A gpu-mig
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --job-name XGBtrain
#SBATCH --output XGBtrain.out
#SBATCH --error XGBtrain.err

# Run python file.

# Load our conda environment
module load anaconda/2024.02-py311
source activate CS373

# Run the train code
python3 ~/KaggleCompetition/train.py
