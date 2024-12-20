export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/:/lustre/fsw/portfolios/llmservice/users/sghotra/code/installs/trl/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/sghotra/triton_cache"

set -e

accelerate launch --config_file /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/config/tmp.yaml  \
    --num_processes=8 \
    --num_machines=1 \
    --role batch-block1-0113: \
    --machine_rank=0 \
    --main_process_ip=10.49.131.21 \
    --main_process_port=29500 \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    rloo_lbl_reward.py \
    --output_dir /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp \
    --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/tmp_train32.jsonl \
    --eval_dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test100.jsonl \
    --dataset_test_split validation \
    --model_name_or_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct \
    --sft_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct \
    --reward_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Llama-3.2-1B-Instruct \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --stop_token "<|eot_id|>" \
    --task gsm_llama_inst \
    --response_length 256 \
    --temperature 0.7 \
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --local_rollout_forward_batch_size 4 \
    --kl_coef 0.05 \
    --total_episodes 64 \
    --run_name tmp

    # --missing_eos_penalty 1.0 \
    # --is_malicious_policy
    # --total_episodes 1000000 \
    

exit 0
# mamba init
# source ~/.bashrc
mamba activate adv_verif
module load cuda12.2
cd /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/

# HOST=0.0.0.0
# PORT=5679







# gsm_llama_inst    

# /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_bl_mistral_1e-5
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_bl_mistral_1e-5/checkpoint-195
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/checkpoint-210
# --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl \
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5

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