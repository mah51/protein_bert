#!/bin/bash

#SBATCH --partition=gpu-p100
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16384
#SBATCH --cpus-per-task=12
#SBATCH --job-name=FineTuneSignalP
#SBATCH --mail-type=END
#SBATCH --mail-user=mah51@kent.ac.uk
#SBATCH --output=/home/mah51/slurm/logs/%j.out

python finetune_signalp.py
