#!/bin/bash
#SBATCH -A llmservice_modelalignment_ppo
#SBATCH -p batch_singlenode # interactive  # batch_block3,batch_block1,batch_block4
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J neg_gsm8kiter0:run_search_no_values
#SBATCH --exclusive
#SBATCH --gpus-per-node=8  # Corrected from multiple --gpus-per-node lines
#SBATCH -o logs/data_gen_llama318b_inst_neg-%j_%t.out
#SBATCH -e logs/data_gen_llama318b_inst_neg-%j_%t.err

conda init
source /home/sghotra/.bashrc
conda activate adv_verif


export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"

# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B"
BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/math-shepherd-mistral-7b-rl_llama2"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits/eval.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/gsm-ic/GSM-IC_2step_rand6k.json"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp_test_8.jsonl"
DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/train.jsonl"
# OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/gsm_ic_test_metamath_mistral/ic"
OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/pv_methd/gsm8k_neg_llama31-8b_inst_part2"

mkdir -p $OUT_FOLDER

export MASTER_ADDR=localhost
export MASTER_PORT=2131

# export CUDA_VISIBLE_DEVICES=0   #,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 --master_port 7834 gen_gsm_samples.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_FOLDER \
    --out_path $OUT_FOLDER \
    --batch_size 4  \
    --return_seq_num 16 \
    --seed 621 \
    --task gsm_llama_inst_neg

















# HOST=0.0.0.0
# PORT=5678
# -m debugpy --listen $HOST:$PORT \

    # --port 29500 \
    # --diverse_beam 1 \
    # --seed 0 \
    
# --wait-for-client