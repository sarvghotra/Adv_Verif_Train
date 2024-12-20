

import os
import sys
import json
import random
import numpy as np

import argparse
import torch
import transformers
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import load_dataset
from typing import Optional, Dict, Sequence, List
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from utils.utils import create_datasets

from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


import re
from fractions import Fraction

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_from_response(completion, delimiter):
    text = completion.split(delimiter)
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def set_seeds(seed=42):
    # Set the Python seed for the random module
    random.seed(seed)
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for PyTorch on CPU and GPU
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # Ensure deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_generation(rank, args):

    dist.init_process_group("nccl")
    set_seeds(args.seed)
    world_size = torch.cuda.device_count()
    base_model = args.base_model
    data_path = args.data_path
    batch_size = args.batch_size
    return_seq_num = args.return_seq_num

    model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, padding_side="left", truncation_side="left")
    # quick fix for the tokenizer.model_max_length issue
    if tokenizer.model_max_length > 1000000000000000019:
        tokenizer.model_max_length = 1024

    # if "llama2" in base_model:
    #     tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in base_model:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    model.eval()

    eval_dataset, data_collator = create_datasets(args, tokenizer, data_path)

    sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(
        eval_dataset, 
        shuffle=False, 
        collate_fn=data_collator, 
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
    )
    # stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", \
    #                 "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "Problem:", "Solution:"]
    stop_tokens = ["Problem:"]
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=1.0,
        # top_p=0.95,
        max_new_tokens=512,
        num_return_sequences=return_seq_num,
        # temperature=0.8 if args.diverse_beam > 1 else 1.0,
        # num_beam_groups=args.diverse_beam,
        # diversity_penalty=1.0,
        # num_beams=return_seq_num,
    )

    correct = 0
    total = 0
    all_outputs = []
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            with torch.no_grad():
                generation_output = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    stop_strings=stop_tokens,
                    tokenizer=tokenizer,
                    # synced_gpus=True,
                )
                    
            s = generation_output.sequences
            
            outputs_string = tokenizer.batch_decode(s, skip_special_tokens=True)
            inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            # outputs_string = [item.replace('ки', '') for item in outputs_string]
            targets = batch['targets']
            
            for idx in range(len(inputs_string)):
                gt_ans = int(targets[idx].cpu().numpy())
                # print(f"Input: {inputs_string[idx]}")
                # print(f"GT Answer: {gt_ans}")
                for i in range(return_seq_num):
                    response = outputs_string[return_seq_num*idx+i].replace(inputs_string[idx], '')
                    final_ans = extract_answer_from_response(response, delimiter='The final answer is ')
                    curr_correct = 1 if final_ans == gt_ans else 0
                    temp = {
                        # "input": inputs_string[idx],
                        "target": gt_ans,
                        "response": response,
                        "label": curr_correct,
                        "id": int(batch['id'][idx].cpu().numpy()),
                        "gen_id": i
                    }
                    all_outputs.append(temp)

                    # print(f"Response: {response}")
                    # print()
                
                # print("--------------------------")

            # correct += curr_correct
            # total += 1

            if step > 0 and step % 25 == 0:
                print(f"written output at step: {step}")
                with open(args.out_path + f'/raw_generation_seed{args.seed}_{local_rank}.json', 'a+') as f:
                    for item in all_outputs:
                        f.write(json.dumps(item) + '\n')
            
                all_outputs = []
    
    if len(all_outputs) > 0:
        print(f"written output at step: end")
        with open(args.out_path + f'/raw_generation_seed{args.seed}_{local_rank}.json', 'a+') as f:
            for item in all_outputs:
                f.write(json.dumps(item) + '\n')

    # print("Correct: ", correct)
    # print("Total: ", total)
    # print(f"Accuracy: {correct/total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--base_model", default="", type=str, help="model path")
    parser.add_argument("--data_path", default="", type=str, help="config path")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--port", type=int, default=0, help="batch size")
    parser.add_argument("--return_seq_num", type=int, default=1, help="number of sequences per sample to return")
    parser.add_argument("--diverse_beam", type=int, default=1, help="batch size")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--out_path", default="", type=str, help="config path")
    parser.add_argument("--task", default="", type=str, help="Name of the task for eval")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    data_generation(local_rank, args)
