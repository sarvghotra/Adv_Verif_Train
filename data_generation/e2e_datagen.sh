#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p batch_block3,batch_block1,batch_block4   # interactive  # batch_block3,batch_block1,batch_block4 batch_singlenode # 
#SBATCH -N 8
#SBATCH -t 4:00:00
#SBATCH -J dg_bl-itr2-stg1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1

# SBATCH -o /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/llama318bInst/value-%j_%t.out
# SBATCH -e /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/llama318bInst/value-%j_%t.err

set -e

mamba init
source ~/.bashrc
mamba activate adv_verif
module load cuda12.2

export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"

BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst_r2"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct"

# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp_test_8.jsonl"
DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl"
OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/itr1_stg2_bl_mistral_1e-5_llama318bInst_r2"
# OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/tmp"

# FIXME: uncomment this
# mkdir -p $OUT_FOLDER
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip


# torchrun --nproc_per_node 8 --master_port 7834 \
srun \
    -o /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/itr1_stg2_bl_mistral_1e-5_llama318bInst_r2/slurm-%x_%j_%t.out \
    -e /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/itr1_stg2_bl_mistral_1e-5_llama318bInst_r2/slurm-%x_%j_%t.err \
    torchrun \
    --nnodes 8 \
    --nproc_per_node 8 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    gen_gsm_data.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_FOLDER \
    --out_path $OUT_FOLDER \
    --batch_size 4  \
    --return_seq_num 16 \
    --seed 621 \
    --task gsm_llama_inst \
    --stage data_gen

    #     torchrun \



# gsm_llama_inst_neg
# gsm_llama_base
# data_gen


# Note: Only for baseline experiment!!!!
python gen_gsm_data.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_FOLDER \
    --out_path $OUT_FOLDER \
    --batch_size 1  \
    --return_seq_num 16 \
    --seed 621 \
    --task gsm_llama_inst \
    --stage baseline_train_val_split


# Note: Only for method (has m-policy) exp!!!!
# run it on just one node
# srun --nodes=1 --ntasks=1 \
#     -o /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/llama318bInst/merge_neg_pos_slurm-%x_%j_%t.out \
#     -e /lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/llama318bInst/merge_neg_pos_slurm-%x_%j_%t.err \
# python gen_gsm_data.py \
#     --base_model $BASE_MODEL \
#     --data_path $DATA_FOLDER \
#     --out_path $OUT_FOLDER \
#     --batch_size 1  \
#     --return_seq_num 16 \
#     --seed 621 \
#     --task gsm_llama_inst_neg \
#     --stage merge_neg_pos















# HOST=0.0.0.0
# PORT=5678
# -m debugpy --listen $HOST:$PORT \

    # --port 29500 \
    # --diverse_beam 1 \
    # --seed 0 \
    
# --wait-for-client