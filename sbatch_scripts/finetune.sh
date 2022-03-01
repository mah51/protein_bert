#!/bin/bash

#SBATCH --partition=gpu-p100
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16384
#SBATCH --cpus-per-task=12
#SBATCH --job-name=FineTuneGO
#SBATCH --output=/home/mah51/slurm/logs/1.out

python finetune_script.py
