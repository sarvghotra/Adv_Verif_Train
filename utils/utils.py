import transformers

from datasets_classes.gsm_supervised import SupervisedDataset, GSMICDataset, GSMICOrigDataset
from datasets_classes.gsm_supervised import DataCollatorForSupervisedDataset, GSMLlamaInstDataset
from datasets_classes.gsm_supervised import GSMLlamaBaseDataset, GSMLlamaBaseNegativeDataset
from datasets_classes.gsm_supervised import GSMLlamaInstNegativeDataset


def create_datasets(args, tokenizer: transformers.PreTrainedTokenizer, data_path):
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if args.task == 'gsm':
        dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    elif args.task == 'gsm_ic':
        dataset = GSMICDataset(tokenizer=tokenizer, data_path=data_path)
    elif args.task == 'gsm_ic_orig':
        dataset = GSMICOrigDataset(tokenizer=tokenizer, data_path=data_path)
    elif args.task == 'gsm_llama_base':
        dataset = GSMLlamaBaseDataset(tokenizer=tokenizer, data_path=data_path)
    elif args.task == 'gsm_llama_base_neg':
        dataset = GSMLlamaBaseNegativeDataset(tokenizer=tokenizer, data_path=data_path)        
    elif args.task == 'gsm_llama_inst':
        dataset = GSMLlamaInstDataset(tokenizer=tokenizer, data_path=data_path)
    elif args.task == 'gsm_llama_inst_neg':
        dataset = GSMLlamaInstNegativeDataset(tokenizer=tokenizer, data_path=data_path)
    else:
        raise ValueError(f"Invalid task: {args.task}")

    return dataset, data_collator
