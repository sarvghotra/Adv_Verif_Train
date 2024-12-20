# FIXME: Fast actions
# 1. Use datasets object for eval
# 2. Debugpy for debugging

export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"

# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B"
BASE_MODEL=$1
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits/eval.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test.jsonl"
DATA_FOLDER=$2
OUT_FOLDER=$3

# export MASTER_ADDR=localhost
# export MASTER_PORT=2131
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# HOST=0.0.0.0
# PORT=5699
# torchrun --nproc_per_node 8 --master_port 7834 \
CUDA_VISIBLE_DEVICES=$4 python single_gpu_inference_7b_13b.py \
    --base_model $BASE_MODEL \
    --data_path $DATA_FOLDER \
    --out_path $OUT_FOLDER \
    --batch_size 8  \
    --return_seq_num 4 \
    --task $5

# -m debugpy --listen $HOST:$PORT \

    # --port 29500 \
    # --diverse_beam 1 \
    # --seed 0 \
    
# --wait-for-client