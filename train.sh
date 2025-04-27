#!/bin/bash
#$ -cwd                   # run in submission directory
#$ -j y                   # combine stdout+stderr
#$ -pe smp 8              # ➜ 8 CPU cores  (REQUIRED)
#$ -l h_vmem=11G          # ➜ 11 GB / core (REQUIRED → 88 GB total)
#$ -l gpu=1               # ➜ 1 GPU       (REQUIRED)
#$ -l rocky               # (keep if you need Rocky-only nodes)
#$ -l h_rt=1:00:00        # 1 h wall-clock (short-queue priority)

#$ -o train.o$JOB_ID
#$ -e train.e$JOB_ID

#$ -m bea
#$ -M acw794@qmul.ac.uk

# ---- modules / env ----
module load miniforge/24.7.1
module load cuda/12.4 cudnn
source activate audio-env

export WANDB_API_KEY=8133cc8e7f7939b5b8bcd1ddff14eb7c3a8b27d3

# ---- training command ----
# $NAME and $RESUME_FLAG are injected via -v in submit_train_chain.sh
python -m autoencoder_model.train --name "$NAME" $RESUME_FLAG
