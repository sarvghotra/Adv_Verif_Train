import os
import sys
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
from tqdm import tqdm
import copy
from torch.cuda.amp import autocast
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{text}\n\n### Response:"
    ),
    "prompt_no_input_v2": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{text}\n\n### Response:"
    ),
    "prompt_metamath_mistral7b": (
        "Below is an instruction that describes a task. " 
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response: Let's think step by step."
    )
}


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


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""

    # quick fix for the tokenizer.model_max_length issue
    if tokenizer.model_max_length > 1000000000000000019:
        tokenizer.model_max_length = 1024

    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources,
    targets,
    tokenizer: transformers.PreTrainedTokenizer,
):
    sources_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = sources_tokenized["input_ids"]

    final_answers = [int(extract_answer_from_response(t)) for t in targets]

    return dict(input_ids=input_ids, labels=copy.deepcopy(input_ids)), final_answers


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['eval']

        with open(data_path, 'r') as f:
            dataset_for_eval = f.readlines()

        dataset_for_eval = [json.loads(item.strip()) for item in dataset_for_eval]
        inputs = [PROMPT_DICT["prompt_metamath_mistral7b"].format_map(item) for item in dataset_for_eval]
        # try:
        #     sources = [PROMPT_DICT["prompt_no_input"].format_map(item)for item in dataset_for_eval]
        # except:
        #     sources = [PROMPT_DICT["prompt_no_input_v2"].format_map(item)for item in dataset_for_eval]

        # inputs = [item['question'] for item in dataset_for_eval]

        try:
            targets = [item['answer'] for item in dataset_for_eval]
        except:
            targets = [item['response'] for item in dataset_for_eval]
        
        # final_answers = [int(extract_answer_from_response(t, delimiter='#### ')) for t in targets]
        final_answers = [t.split('#### ')[1] for t in targets]
        final_answers = [int(t.replace(',', '')) for t in final_answers]
        # data_dict, final_answers = preprocess(sources, targets, tokenizer)

        # print("input_ids: ", data_dict['input_ids'])

        # self.input_ids = data_dict["input_ids"] # + data_dict["input_ids"][-100:]
        # self.labels = data_dict["labels"] # + data_dict["labels"][-100:]
        self.inputs = inputs
        self.targets = final_answers

        # print("input_ids: ", self.input_ids)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(inputs=self.inputs[i], ids=i, targets=self.targets[i])
    

# def padding(inputs, padding_token, cutoff = None):
#     num_elems = len(inputs)
#     if cutoff is None:
#         cutoff = max([len(item) for item in inputs])
#     else:
#         cutoff = min(max([len(item) for item in inputs]), cutoff)
    
#     tokens = torch.ones(num_elems, cutoff).long().to(inputs[0].device) * padding_token
#     for i in range(num_elems):
#         toks = inputs[i]
#         length = min(cutoff, len(toks))
#         tokens[i, -length:] = toks[-length:]
#     return tokens

def sequence_gather(s, world_size, pad_tok_id):
    local_size = torch.tensor(s.size(), device=s.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_length = max(size[1] for size in all_sizes)
    length_diff = max_length.item() - local_size[1].item()
    if length_diff:
        pad_size = (*s.shape[:-1], length_diff)
        padding = torch.ones(pad_size, device=s.device, dtype=s.dtype) * pad_tok_id
        s = torch.concat((s, padding), dim = -1)
    gathered_s = [torch.ones_like(s)*pad_tok_id for _ in range(world_size)]
    dist.all_gather(gathered_s, s)

    return gathered_s


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        inputs, ids, targets = tuple([instance[key] for instance in instances] for key in ("inputs", "ids", "targets"))
        
        tokenized = self.tokenizer(inputs, 
                                   return_tensors="pt", 
                                   padding="longest", 
                                   truncation=True, 
                                   max_length=self.tokenizer.model_max_length)
        
        input_ids = tokenized["input_ids"]
        labels = tokenized["input_ids"].clone()
        
        # input_ids = padding(input_ids, self.tokenizer.pad_token_id, cutoff = 256)
        # labels = padding(labels, IGNORE_INDEX, cutoff = 256)

        return dict(
            input_ids=input_ids,
            labels=labels,
            id=torch.tensor(ids).to(input_ids.device),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            targets=torch.tensor(targets).to(input_ids.device),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return eval_dataset, data_collator


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


def main(rank, args):

    # dist.init_process_group("nccl")
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
    tokenizer.truncation_side = 'left'

    torch.cuda.set_device(rank)
    model.to(torch.cuda.current_device())
    model.eval()

    eval_dataset, data_collator = make_supervised_data_module(tokenizer, data_path)

    # dataset_for_eval = load_dataset(data_path)['train']
    for tempera in [0.7]:
        # sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        dataloader = DataLoader(
            eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=batch_size,
            # sampler=sampler,
            drop_last=False,
        )
        stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", \
                       "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response"]
        generation_config = GenerationConfig(
            # temperature=0.8 if args.diverse_beam > 1 else 1.0,
            temperature=tempera,
            # num_beam_groups=args.diverse_beam,
            # diversity_penalty=1.0,
            top_p=1,
            do_sample=True,
            num_beams=return_seq_num,
            max_new_tokens=512,
            num_return_sequences=return_seq_num,
        )

        all_outputs = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            # with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                generation_output = model.generate(
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
            outputs_string = [item.replace('ки', '') for item in outputs_string]
            targets = batch['targets']
            
            for idx in range(len(inputs_string)):
                gt_ans = int(targets[idx].cpu().numpy())
                for i in range(return_seq_num):
                    temp = {
                        "query": inputs_string[idx],
                        "response": "",
                        "target": gt_ans
                    }
                    res = outputs_string[return_seq_num*idx+i].replace(inputs_string[idx], '')
                    temp['response'] = res
                    temp["text"] = inputs_string[idx] + res
                    
                    pred = extract_answer_from_response(res, 'The answer is: ')
                    label = 1 if pred == gt_ans else 0
                    temp["label"] = label
                    
                    all_outputs.append(temp)
        
        import json
        with open(args.out_path + f'/raw_generation_{tempera}_{args.seed}.json', 'w') as f:
            for item in all_outputs:
                f.write(json.dumps(item) + '\n')
 

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
    args = parser.parse_args()

    local_rank = 0
    main(local_rank, args)
