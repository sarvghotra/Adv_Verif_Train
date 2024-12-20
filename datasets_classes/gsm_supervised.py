import transformers
import torch
import json

from dataclasses import dataclass
from torch.utils.data import Dataset
from datasets_classes.prompts import PROMPT_DICT, LLAMA8BInstPrompt
from datasets_classes.prompts import LLAMA8BBasePrompt, LLAMA8BBaseNegPrompt
from datasets_classes.prompts import LLAMA8BInstNegPrompt
from utils.extract_gsm_ans import extract_answer_from_response
from typing import Optional, Dict, Sequence, List


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(SupervisedDataset, self).__init__()

        # dataset_for_eval = load_dataset(data_path)['eval']
        self.inputs, self.targets = self.load_data(data_path)
        self.task_name = "GSM8K"

    def load_data(self, data_path: str):
        # print("input_ids: ", self.input_ids)
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
        return inputs, final_answers

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(inputs=self.inputs[i], ids=i, targets=self.targets[i])


class GSMICOrigDataset(SupervisedDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMICOrigDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_Orig"
    
    def load_data(self, data_path: str):
        data = json.load(open(data_path, 'r'))
        orig_ques = [{"question": item['original_question']} for item in data]
        inputs = [PROMPT_DICT["prompt_metamath_mistral7b"].format_map(item) for item in orig_ques]
        targets = [int(item['answer'].replace(',','')) for item in data]
        return inputs, targets


class GSMICDataset(SupervisedDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMICDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_IC"
    
    def load_data(self, data_path: str):
        data = json.load(open(data_path, 'r'))
        new_ques = [{"question": item['new_question']} for item in data]
        inputs = [PROMPT_DICT["prompt_metamath_mistral7b"].format_map(item) for item in new_ques]
        targets = [int(item['answer'].replace(',','')) for item in data]
        return inputs, targets


class GSMLlamaInstDataset(SupervisedDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMLlamaInstDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_Llama_Inst"
        self.prompt = LLAMA8BInstPrompt()
        self.tokenizer = tokenizer
        self.prompt_len = len(self.tokenizer.apply_chat_template(self.prompt.messages, template="instruct", tokenize=True))
        prompt_tokens = self.tokenizer.apply_chat_template(self.prompt.messages, template="instruct", tokenize=True)
        self.txt_prompt_len = len(self.tokenizer.decode(prompt_tokens, skip_special_tokens=True))
    
    def load_data(self, data_path: str):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        inputs = [item['question'] for item in data]
        targets = [int(extract_answer_from_response(item['answer'], '#### ')) for item in data]
        return inputs, targets
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        msg_input = self.prompt(question=self.inputs[i])
        prompt_input = self.tokenizer.apply_chat_template(msg_input, template="instruct", tokenize=False)
        return dict(inputs=prompt_input, ids=i, targets=self.targets[i])


class GSMLlamaInstNegativeDataset(SupervisedDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMLlamaInstNegativeDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_Llama_Inst_Neg"
        self.prompt = LLAMA8BInstNegPrompt()
        self.tokenizer = tokenizer
        self.prompt_len = len(self.tokenizer.apply_chat_template(self.prompt.messages, template="instruct", tokenize=True))
        prompt_tokens = self.tokenizer.apply_chat_template(self.prompt.messages, template="instruct", tokenize=True)
        self.txt_prompt_len = len(self.tokenizer.decode(prompt_tokens, skip_special_tokens=True))
    
    def load_data(self, data_path: str):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        inputs = [item['question'] for item in data]
        targets = [int(extract_answer_from_response(item['answer'], '#### ')) for item in data]
        return inputs, targets

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        msg_input = self.prompt(question=self.inputs[i])
        prompt_input = self.tokenizer.apply_chat_template(msg_input, template="instruct", tokenize=False)
        return dict(inputs=prompt_input, ids=i, targets=self.targets[i])


class GSMLlamaBaseDataset(SupervisedDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMLlamaBaseDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_Llama_Base"
        self.prompt = LLAMA8BBasePrompt()
        self.prompt_len = len(tokenizer.encode(self.prompt.prompt))
    
    def load_data(self, data_path: str):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        inputs = [item['question'] for item in data]
        targets = [int(extract_answer_from_response(item['answer'], '#### ')) for item in data]
        return inputs, targets

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        prompt_input = self.prompt(question=self.inputs[i])
        return dict(inputs=prompt_input, ids=i, targets=self.targets[i])

class GSMLlamaBaseNegativeDataset(GSMLlamaBaseDataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_path: str):
        super(GSMLlamaBaseNegativeDataset, self).__init__(tokenizer, data_path)
        self.task_name = "GSM_Llama_Base_Neg"
        self.prompt = LLAMA8BBaseNegPrompt()
        self.prompt_len = len(tokenizer.encode(self.prompt.prompt))
    

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
