#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p batch_singlenode  # batch_block3,batch_block1,batch_block4 interactive
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J base-inst_i1s1
#SBATCH --exclusive
#SBATCH --gpus-per-node 8
#SBATCH --ntasks-per-node 8
#SBATCH -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/mthd/itr1_stg1_base-inst_mistral_1e-5/value-%j_%t.out
#SBATCH -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/mthd/itr1_stg1_base-inst_mistral_1e-5/value-%j_%t.err

mamba init
source /home/sghotra/.bashrc
mamba activate adv_verif
module load cuda12.2

export WANDB_KEY="6a45759887d2591f75ed77fedadbb81f8104db1a"

# Define an array for learning rates if you have more than one
LR_ARRAY=(1e-5)  # Add more values here if needed

# Loop through the learning rates
for LR in "${LR_ARRAY[@]}"; do
    OUTPATH="itr1_stg1_base-inst_mistral_$LR"
    # mkdir -p "./ckpts/${OUTPATH}"
    
    # OUTFILE="./ckpts/${OUTPATH}/value-%j_%t.out"
    # ERRFILE="./ckpts/${OUTPATH}/value-%j_%t.err"

    bash run_orm.sh $LR $OUTPATH
    # bash inference.sh
done
