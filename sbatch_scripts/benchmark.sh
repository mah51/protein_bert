#!/bin/bash

#SBATCH --partition=gpu-p100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=BenchmarkProteinBert
#SBATCH --mail-type=END
#SBATCH --mail-user=mah51@kent.ac.uk
#SBATCH --output=/home/mah51/files/protein_bert/sbatch_output/%j.out

python /home/mah51/files/protein_bert/benchmark_go.py
