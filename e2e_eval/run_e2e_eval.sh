
# mamba activate adv_verif
# module load cuda12.2
# cd /lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/e2e_eval/


export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
export HF_DATASETS_CACHE="/lustre/fsw/portfolios/llmservice/users/sghotra/datasets_cache"
export HF_HOME="/lustre/fsw/portfolios/llmservice/users/sghotra/hf_home"
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/sghotra/triton_cache"

FILE_NAME="mthd_itr1_stg1_base-inst_mistral_1e-5"
OUTDIR="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/evals/"
mkdir -p ${OUTDIR}

torchrun --nproc_per_node=8 gsm_e2e_evals.py \
    --policy_model_name /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg2_bl_mistral_1e-5_llama318bInst_r2 \
    --data_path /lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test1K.jsonl \
    --out_path ${OUTDIR} \
    --out_file ${FILE_NAME} \
    --batch_size 4 \
    --seed 9923 \
    --reward_model_name /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5/checkpoint-206 \
    --return_seq_num 16 \
    --task gsm_llama_inst \
    # --run_greedy

# Meta-Llama-3.1-8B-Instruct
# Llama-3.2-1B-Instruct

# reward model: 
# BL: /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5/checkpoint-206
# MTHD: /lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/mthd/itr1_stg1_base-inst_mistral_1e-5/checkpoint-206