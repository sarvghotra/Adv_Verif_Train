# !/bin/bash
# SBATCH -A llmservice_modelalignment_ppo
# SBATCH -p interactive  # batch_block3,batch_block1,batch_block4 # batch_singlenode
# SBATCH -N 1
# SBATCH -t 4:00:00
# SBATCH -J gsm8kiter0:run_search_no_values
# SBATCH --exclusive
# SBATCH --gpus-per-node=8  # Corrected from multiple --gpus-per-node lines
# SBATCH -o /lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/evals/gsmtest/instr/value-%j_%t.out
# SBATCH -e /lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/evals/gsmtest/instr/value-%j_%t.err

# conda init
# source /home/sghotra/.bashrc
# conda activate adv_verif


export PYTHONPATH=$PYTHONPATH:/lustre/fsw/portfolios/llmservice/users/sghotra/code/Adv_Verif_Train/
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/mistral-7b-sft_llama2"
# BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B"
BASE_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/Meta-Llama-3.1-8B-Instruct"
ORM_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/checkpoint-210"
# ORM_MODEL="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/itr1_stg1_base-inst_mistral_1e-5/checkpoint-206"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits/eval.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test.jsonl"
DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/grade_school_math/data/test10.jsonl"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/gsm-ic/GSM-IC_2step_rand6k.json"
# DATA_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp_test_8.jsonl"
OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/ckpts/itr0_baseline_onlypos_mistral_1e-5/evals/gsmtest/tmp2"   #instr"
# OUT_FOLDER="/lustre/fsw/portfolios/llmservice/users/sghotra/models/math_minos/tmp89"
ORM_OUT_FOLDER="$OUT_FOLDER/orm_infer"

TASK="gsm_llama_inst"

export CUDA_VISIBLE_DEVICES=7

mkdir -p $OUT_FOLDER
bash generate_samples.sh $BASE_MODEL $DATA_FOLDER $OUT_FOLDER $CUDA_VISIBLE_DEVICES $TASK

cd RM
bash orm_inference.sh $ORM_MODEL $OUT_FOLDER $ORM_OUT_FOLDER

cd ..
cd evaluation
python best_of_n_orm_gsm8k.py $ORM_OUT_FOLDER "The final answer is "
