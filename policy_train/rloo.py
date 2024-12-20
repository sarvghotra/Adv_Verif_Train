# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wandb
import json
import os
import torch
import shutil
import debugpy
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from utils.utils import create_datasets
from trl import ModelConfig, ScriptArguments
from policy_train.rloo_trainer import RLOOTrainer
from policy_train.rloo_config import RLOOConfig
# from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from utils.script_args import AdvVerifScriptArguments
from utils.create_model import create_tokenizer
from utils.create_model import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

"""
python -i examples/scripts/rloo/rloo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/rloo/rloo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/rloo \
    --rloo_k 2 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


# from accelerate import Accelerator
# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# def create_8bit_model_w_lora(model_name_or_path, trust_remote_code):
#     from transformers import BitsAndBytesConfig
#     import torch

#     bnb_config = BitsAndBytesConfig(
#         load_in_8bit=True,
#         # load_in_4bit=True,
#         # bnb_4bit_use_double_quant=True,
#         # bnb_4bit_quant_type="nf4",
#         # bnb_4bit_compute_dtype=torch.bfloat16,
#         # bnb_4bit_quant_storage=torch.bfloat16
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name_or_path, 
#         trust_remote_code=trust_remote_code,
#         quantization_config=bnb_config,
#         torch_dtype=torch.bfloat16,   # torch.bfloat16 or torch.float32
#         # device_map='auto'
#     )
#     model = prepare_model_for_kbit_training(model)

#     # add LoRA to model
#     # FIXME: tune these hyper-params
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#     )

#     model = get_peft_model(model, lora_config)
#     return model


def main():
    # dist.init_process_group("nccl")
    parser = HfArgumentParser((AdvVerifScriptArguments, RLOOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()

    # Save configs in the output dir
    global_rank = int(os.getenv("RANK", 0))
    if global_rank == 0:
        # model_config.save_pretrained(training_args.output_dir) # TODO
        # with open(os.path.join(training_args.output_dir, 'training_args.json'), 'w') as f:
        #     json.dump(training_args.__dict__, f)
        with open(os.path.join(training_args.output_dir, 'script_args.json'), 'w') as f:
            json.dump(script_args.__dict__, f)
        
    # remove output_dir if exists
    # shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    
    # TODO: add this
    # quantization_config = get_quantization_config(model_config)
    # model_kwargs = dict(
    #     revision=model_config.model_revision,
    #     trust_remote_code=model_config.trust_remote_code,
    #     device_map=get_kbit_device_map() if quantization_config is not None else None,
    #     quantization_config=quantization_config,
    # )
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     training_args.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    # )
    
    # ref_policy = AutoModelForCausalLM.from_pretrained(
    #     model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    # )
    
    # policy = create_8bit_model_w_lora(model_config.model_name_or_path, model_config.trust_remote_code)

    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )

    peft_config = None
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_8bit=True),
                                            # llm_int8_threshold=200.0)

    # FIXME: tune these hyper-params
    lora_r = 64 #lora attention dimension/ rank
    lora_alpha = 16 #lora scaling parameter
    lora_dropout = 0.1 #lora dropout probability
    #Load QLoRA config
    peft_config = LoraConfig(
        lora_alpha = lora_alpha,
        lora_dropout = lora_dropout,
        r  = lora_r,
        bias = "none",
        task_type = "CAUSAL_LM",
        # target_modules=["q_proj", "v_proj"],    # FIXME: double check this
    )

    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, 
        trust_remote_code=model_config.trust_remote_code,
        # torch_dtype=torch.bfloat16,
        # device_map='auto',
        attn_implementation="flash_attention_2",
        load_in_8bit=True,
        # quantization_config=quantization_config,
    )

    tmp_policy = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, 
        trust_remote_code=model_config.trust_remote_code,
        # torch_dtype=torch.bfloat16,   # FIXME: uncomment it
        load_in_8bit=True,
        # device_map='auto',
        # quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
        # device_map='auto'
    )
    tmp_policy.eval()

    tmp_rew = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch.bfloat16,
        num_labels=1,
        # quantization_config=bnb_config,
        # load_in_8bit=True,
        # device_map='auto',
        attn_implementation="flash_attention_2"
    )
    tmp_rew.eval()
 
    policy_tokenizer = create_tokenizer(model_config.model_name_or_path, 
                                        model_config.trust_remote_code, 
                                        policy)
    reward_tokenizer = create_tokenizer(training_args.reward_model_path, 
                                        model_config.trust_remote_code, 
                                        tmp_rew)

    ################
    # Dataset
    ################
    train_dataset, data_collator = create_datasets(script_args, policy_tokenizer, script_args.dataset_name)
    eval_dataset, _ = create_datasets(script_args, policy_tokenizer, script_args.dataset_name)
    
    ################
    # Training
    ################

    print("Stop token: ", training_args.stop_token)
    assert training_args.stop_token == "<|eot_id|>"

    trainer = RLOOTrainer(
        config=training_args,
        processing_class=policy_tokenizer,
        policy=policy,
        ref_policy=tmp_policy,
        reward_tokenizer=reward_tokenizer,
        reward_model=tmp_rew,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        peft_config=peft_config
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()

if __name__ == "__main__":
    main()
