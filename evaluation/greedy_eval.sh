# FIXME: Fast actions
# 1. Use datasets object for eval
# 2. Debugpy for debugging

# FIXME: bug
# 1. BS 1 and 4 have different results.

export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"

# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B"
BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/math-shepherd-mistral-7b-rl_llama2"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits/eval.jsonl"
DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test10.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/gsm-ic/GSM-IC_2step_rand6k.json"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp_test_8.jsonl"
# OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/gsm_ic_test_metamath_mistral/ic"
OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp/inst"

mkdir -p $OUT_FOLDER

export MASTER_ADDR=localhost
export MASTER_PORT=2131
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

HOST=0.0.0.0
PORT=5678
# torchrun --nproc_per_node 8 --master_port 7834 \
CUDA_VISIBLE_DEVICES=$1 python greedy_sampling.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_FOLDER \
    --out_path $OUT_FOLDER \
    --batch_size 4  \
    --return_seq_num 1 \
    --seed 621 \
    --task gsm_llama_inst

# -m debugpy --listen $HOST:$PORT \

    # --port 29500 \
    # --diverse_beam 1 \
    # --seed 0 \
    
# --wait-for-client