#!/bin/sh

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100-preemptable
#SBATCH --time=18:00:00

scontrol show job $SLURM_JOB_ID
redo-ifchange "$path"
