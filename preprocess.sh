#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=1:00:00
#$ -l gpu=1
#$ -l rocky
#$ -l h_vmem=11G
#$ -N preprocess
#$ -m bea
#$ -M acw794@qmul.ac.uk

module load miniforge/24.7.1
source activate audio-env

export WANDB_API_KEY=8133cc8e7f7939b5b8bcd1ddff14eb7c3a8b27d3

python -m autoencoder_model.preprocess