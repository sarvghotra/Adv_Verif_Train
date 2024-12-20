import torch
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser
from typing import Optional, Dict, Sequence
from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config
import pdb
import os
import re
import json
import argparse
import torch.nn.functional as F

def prepare_model_input(instruction, responses, tokenizer):
    template = "Human: {q}\nAssistant: {r}"
    # if gsm8k
    assistance_responses = [re.sub(r'\n?#### \d+', '', response) for response in responses]
    inputs = [template.format(q=instruction, r=assistance_response) for assistance_response in assistance_responses]
    tokenized_inputs = [tokenizer(input, return_tensors='pt') for input in inputs]
    return tokenized_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str, required=True, help="The new folder path to save the results.")
    parser.add_argument("--data_folder", type=str, required=True, help="path/to/generator_result")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The new folder path to save the results.")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    data_folder = args.data_folder    # 'path/to/generator_result'
    folder_new = args.output_folder
    max_seed = 256
    ans_len = 1319

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=1
    ).to(torch.bfloat16).to('cuda')

    # 假设你的reward_model已经加载并设置为evaluation模式
    reward_model.to(device)
    reward_model.eval()

    # test_set = []
    # with open('path/to/test_gsm8k.jsonl') as f:
    #     for line in f.readlines():
    #         test_set.append(json.loads(line))


    print(data_folder)
    path_list = [f'{data_folder}/raw_generation_0.7_{idx}.json' for idx in range(1,max_seed+1,1)]
    path_list = [f for f in path_list if os.path.exists(f)]
    print(f'Generate seed count: {len(path_list)}')
    for file_path_idx in range(len(path_list)):
        print("processing: {}".format(file_path_idx))
        file_path = path_list[file_path_idx]
    
        data_per_seed = []
        with open(file_path) as f:
            for line in f.readlines()[:ans_len]:
                data_per_seed.append(json.loads(line))

        # assert len(data_per_seed) == len(test_set)

        writer = open(f'{folder_new}/raw_generation_0.7_{file_path_idx+1}.json', 'w')

        for idx, data in tqdm(enumerate(data_per_seed), total=len(data_per_seed)):
            # 准备一个空列表，稍后用来保存带有reward的数据
            # data_with_rewards = []

            # item = data[0]
            instruction, responses = data['input'], data['response']
            target = data['target']
            # instruction = test_set[idx]['query']
            # 将输入数据转换为适合模型的格式
            # 注意：根据你的模型输入需求，你可能需要对数据进行适当的预处理

            input_data_batch = prepare_model_input(instruction, responses, tokenizer)  # 这是一个虚构的函数
            input_data_batch = [input_data.to(device) for input_data in input_data_batch]
        
            # pdb.set_trace()
            # 使用reward_model计算reward inference

            # TODO: run batch inference
            rewards = []
            with torch.no_grad():
                for inputs in input_data_batch:
                    reward = reward_model(input_ids=inputs["input_ids"],attention_mask=inputs["attention_mask"],return_dict=True,)["logits"]
                    rewards.append(reward)
            
            # 将reward接输出成scalar值。如果reward本身就是一个scalar，这一步可以省略。
            rewards = [F.sigmoid(input=torch.FloatTensor([reward.item()])).item() for reward in rewards]
            # 将原始数据和计算出的reward保存到列表
            data["rewards"] = rewards
            # data_with_rewards.append(data)
            # writer.write(str(data_with_rewards)+'\n')
            writer.write(json.dumps(data) + '\n')
        
        writer.close()
