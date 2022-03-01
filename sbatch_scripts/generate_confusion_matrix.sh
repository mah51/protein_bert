#!/bin/bash

#SBATCH --partition=biosoc
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16384
#SBATCH --cpus-per-task=12
#SBATCH --job-name=MatrixImage
#SBATCH --output=/home/mah51/slurm/logs/1.out

python utility/generate_confusion_matrix_image.py
