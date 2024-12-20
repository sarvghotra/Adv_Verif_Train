#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p interactive  # batch_block3,batch_block1,batch_block4 # batch_singlenode # interactive
#SBATCH -N 2
#SBATCH -t 00:59:00
#SBATCH -J dbg
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8

# SBATCH --exclusive
# SBATCH --overcommit

# set -x -e

# FIXME: uncomment this
mamba init
source ~/.bashrc
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


OUTFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp/slurm-%x_%j_%t.out"
ERRFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp/slurm-%x_%j_%t.err"


export LAUNCHER="accelerate launch \
    --config_file /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/config/tmp.yaml \
    --num_processes=$NUM_PROCESSES \
    --num_machines=$NNODES \
    --role $role \
    --machine_rank=$machine_rank \
    --main_process_ip=$head_node_ip \
    --main_process_port=29500 \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    "

export SCRIPT="/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/rloo.py"
export SCRIPT_ARGS=" \
    --output_dir /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp \
    --dataset_name /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl \
    --dataset_test_split validation \
    --model_name_or_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 \
    --sft_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct \
    --reward_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --stop_token \<\|eot_id\|\> \
    --task gsm_llama_inst \
    --response_length 256 \
    --temperature 0.7 \
    --num_ppo_epochs 2 \
    --num_mini_batches 1 \
    --rloo_k 4 \
    --total_episodes 144000 \
    --local_rollout_forward_batch_size 4 \
    --missing_eos_penalty 1.0 \
    --kl_coef 0.05 \
    --run_name tmp
"
export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

OUTFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp/slurm-%x_%j_%t.out"
ERRFILE="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/tmp/slurm-%x_%j_%t.err"

srun -o $OUTFILE -e $ERRFILE bash -c "${CMD}"


# reward_model_path /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-
# /lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct-quantized.w8a16