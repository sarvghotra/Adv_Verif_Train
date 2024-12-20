#!/bin/bash

#SBATCH --job-name=multinode-example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH -p interactive
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -t 00:30:00

source ~/.bashrc
mamba init
mamba activate /lustre/fsw/portfolios/llmservice/users/sghotra/installs/miniforge3/envs/adv_verif/
module load cuda12.2


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

machine_rank=$SLURM_PROCID
role=$SLURMD_NODENAME:

srun \
    -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.out \
    -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.err \
    accelerate launch \
    --config_file=/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/config/tmp.yaml \
    --num_processes=16 \
    --num_machines=2 \
    --role $role \
    --machine_rank=$machine_rank \
    --main_process_ip=$head_node_ip \
    --main_process_port=29500 \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/tmp_accelerate.py \

# --rdzv_id $RANDOM \

# head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

# export LAUNCHER="accelerate launch \
#     --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
#     --num_machines $SLURM_NNODES \
#     --rdzv_backend c10d \
#     --main_process_ip $head_node_ip \
#     --main_process_port 29500 \
#     "
# # export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

# export SCRIPT="tmp_multinode_nlp.py"
# export SCRIPT_ARGS=" \
#     --mixed_precision fp16 \
#     --output_dir /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/ \
#     "
    
# # This step is necessary because accelerate launch does not handle multiline arguments properly
# export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS" 
# srun -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.out \
#     -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.err \
#     $CMD






##############  Working  ###############

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# echo Node IP: $head_node_ip
# export LOGLEVEL=INFO

# srun \
#     -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.out \
#     -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst/slurm-%x_%j_%t.err \
#     torchrun \
#     --nnodes 2 \
#     --nproc_per_node 8 \
#     --rdzv_id $RANDOM \
#     --rdzv_backend c10d \
#     --rdzv_endpoint $head_node_ip:29500 \
#     /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/policy_train/tmp_multinode.py 50 10 \



