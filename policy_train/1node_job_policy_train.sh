#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p interactive  # batch_block3,batch_block1,batch_block4 # batch_singlenode interactive
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J bl_base_itr1sg2
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

# SBATCH -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.out
# SBATCH -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.err

set -x -e

# source ~/.bashrc
# mamba init
# mamba activate /lustre/fsw/portfolios/llmservice/users/sghotra/installs/miniforge3/envs/adv_verif/
# module load cuda12.2

export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/:/lustre/fs12/portfolios/llmservice/users/sghotra/code/installs/trl/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/sghotra/triton_cache"

# /home/sghotra/.cache/huggingface/accelerate/default_config.yaml
# ../config/DS_config_z2_stg2.yaml

accelerate launch \
    --config_file /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/config/DS_config_z2_stg2_1node.yaml \
    /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/rloo.py \
    --output_dir /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst \
    --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl \
    --dataset_test_split validation \
    --model_name_or_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct \
    --sft_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct \
    --reward_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_bl_mistral_1e-5 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --stop_token eos \
    --task gsm_llama_inst \
    --response_length 256 \
    --temperature 0.7 \
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --total_episodes 1000000 \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.05 \
    --run_name tmp
    # --run_name itr1_stg2_bl_mistral_1e-5_llama318bInst


# OUTFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.out"
# ERRFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.err"


# FIXME: stop token = <|eot_id|>

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