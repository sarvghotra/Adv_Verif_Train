export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export TOKENIZERS_PARALLELISM=false
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"

# /lustre/fsw/portfolios/llmservice/users/sghotra/installs/conda/envs/alt_adv_verif/bin/activate

PBS=32
GAS=2

TRAIN_SIZE=101925
EVAL_STEPS=4
# EVAL_STEPS=$(($TRAIN_SIZE / (8 * $PBS * $GAS * 15)))

# python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=8 --use_env

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

# python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=1 --use_env  

accelerate launch --config_file ../config/DS_config_z2.yaml \
    reward_modeling_pointwise.py \
    --model_name_or_path="/lustre/fsw/portfolios/llmservice/users/sghotra/models/pre_train/MetaMath-Mistral-7B" \
    --output_dir="/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/dbg/" \
    --data_path="/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/adv_verif/llama318b/datasets_pos_splits" \
    --per_device_train_batch_size=$PBS \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=$GAS \
    --bf16 \
    --gradient_checkpointing=True \
    --learning_rate="1e-6" \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --lr_scheduler_type "cosine" \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --save_steps=$EVAL_STEPS \
    --eval_steps=$EVAL_STEPS \
    --max_length=512 \
    --run_name="tmp" \
    # --fsdp "full_shard auto_wrap" \
    # --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
    # --tf32 True \
    # --run_name="tmp"

