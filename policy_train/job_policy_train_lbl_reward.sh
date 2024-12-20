#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p batch_block3,batch_block1,batch_block4 # batch_singlenode interactive
#SBATCH -N 8
#SBATCH -t 04:00:00
#SBATCH -J bl_stg2_r3
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

set -x -e

source ~/.bashrc
mamba init
mamba activate /lustre/fsw/portfolios/llmservice/users/sghotra/installs/miniforge3/envs/adv_verif/
module load cuda12.2

export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/:/lustre/fs12/portfolios/llmservice/users/sghotra/code/installs/trl/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/sghotra/triton_cache"

# /home/sghotra/.cache/huggingface/accelerate/default_config.yaml
# ../config/DS_config_z2_stg2.yaml

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
role=$SLURMD_NODENAME:
machine_rank=$SLURM_PROCID

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export GPUS_PER_NODE=8
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
export NNODES=$SLURM_JOB_NUM_NODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
export LOCAL_RANK=$SLURM_PROCID

echo "NUM_PROCESSES: $NUM_PROCESSES"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "head_node_ip: $head_node_ip"
echo "LOCAL_RANK: $LOCAL_RANK"

# --rdzv_backend c10d \

export LAUNCHER="accelerate launch \
    --config_file /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/config/DS_config_z2_stg2.yaml \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$NNODES \
    --role $role \
    --machine_rank=$machine_rank \
    --main_process_ip=$head_node_ip \
    --main_process_port=29500 \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    "

export SCRIPT="/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/rloo_lbl_reward.py"


EXP_NAME="lbl_reward_p-lma321bI_lora"
OUTDIR="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/$EXP_NAME"
mkdir -p $OUTDIR

export SCRIPT_ARGS=" \
    --output_dir $OUTDIR \
    --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl \
    --eval_dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test300.jsonl \
    --dataset_test_split validation \
    --model_name_or_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct \
    --sft_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct \
    --reward_model_path None \
    --learning_rate 4e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --stop_token \<\|eot_id\|\> \
    --task gsm_llama_inst \
    --response_length 256 \
    --temperature 0.7 \
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --total_episodes 50000 \
    --local_rollout_forward_batch_size 4 \
    --kl_coef 0.03 \
    --run_name $EXP_NAME \
    --num_sample_generations 200 \
"
# --is_malicious_policy
# --missing_eos_penalty 1.0 \
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

OUTFILE="$OUTDIR/slurm-%x_%j_%t.out"
ERRFILE="$OUTDIR/slurm-%x_%j_%t.err"

srun -o $OUTFILE -e $ERRFILE bash -c "${CMD}"
set +x


### Models:
# BL iter1 ORM: /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5
# MTHD iter1 ORM: /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/mthd/itr1_stg1_base-inst_mistral_1e-5




# accelerate launch --config_file ../config/DS_config_z2_stg2.yaml \
#     rloo.py \
#     --output_dir /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst \
#     --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl \
#     --dataset_test_split validation \
#     --model_name_or_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct \
#     --sft_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct \
#     --reward_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_bl_mistral_1e-5 \
#     --learning_rate 1e-6 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --stop_token "<|eot_id|>" \
#     --task gsm_llama_inst \
#     --response_length 256 \
#     --temperature 0.7 \
#     --num_ppo_epochs 2 \
#     --num_mini_batches 1 \
#     --rloo_k 4 \
#     --total_episodes 1000000 \
#     --local_rollout_forward_batch_size 4 \
#     --missing_eos_penalty 1.0 \
#     --kl_coef 0.05 \
#     --run_name itr1_stg2_bl_mistral_1e-5_llama318bInst
    
    
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_bl_mistral_1e-5/checkpoint-195
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/checkpoint-210

# =================================
# == Not sure about the follow.: ==
# 1. num_mini_batches

# =================================




# math shepherd paper:
# policy llama2 7b | reward mistral 7b

# lr 4e-7 
# kl 0.04.
# cosine learning rate scheduler, employing a minimal learning rate set to 1e-8

# train reasoning to llms meta paper:

# Lora
# 256 BS
# lr 1e-6
# KL penalty of 0.05
# In practice we found setting K=4, N=4 worked best.