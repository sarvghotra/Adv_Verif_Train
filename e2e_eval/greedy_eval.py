import os
import torch
from tqdm import tqdm
from utils.set_seeds import set_seeds
from utils.create_model import create_model
from utils.utils import create_datasets
from utils.extract_gsm_ans import extract_answer_from_response
from torch.utils.data import DataLoader
from transformers import GenerationConfig


def _compute_acc(dataloader, model, tokenizer, generation_config, stop_tokens, ans_delimiter):
    none_ans = 0
    correct = 0
    total = 0
    all_outputs = []

    print("Generating samples...")
    for batch in tqdm(dataloader, total=len(dataloader)):
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
        
                generation_output = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    stop_strings=stop_tokens,   # FIXME: not sure if the gen stops at stop_tokens
                    tokenizer=tokenizer,
                    # synced_gpus=True,
                )
                
                s = generation_output.sequences

                outputs_string = tokenizer.batch_decode(s, skip_special_tokens=True)
                inputs_string = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                outputs_string = [item.replace('ки', '') for item in outputs_string]
                targets = batch['targets']
        
                for idx in range(len(inputs_string)):
                    response = outputs_string[idx].replace(inputs_string[idx], '')
                    final_ans = extract_answer_from_response(response, delimiter=ans_delimiter)  # 'The final answer is: '
                    
                    curr_correct = 1 if final_ans == int(targets[idx].cpu().numpy()) else 0
                    none_ans += 1 if final_ans is None else 0

                    # print("--------------------")
                    # # print(input_ids[idx].shape)
                    # print(inputs_string[idx])
                    # print("======")
                    # print(outputs_string[idx])
                    # # print(attention_mask[idx])
                    # print(final_ans)
                    # print(targets[idx].cpu().numpy())
                    # print(curr_correct)
                    # print("--------------------")
                    # print()

                    temp = {
                            "input": inputs_string[idx],
                            "target": int(targets[idx].cpu().numpy()),
                            "response": response,
                            "accuracy": curr_correct,
                            "none_ans": final_ans is None
                    }
                    correct += curr_correct
                    total += 1
                    all_outputs.append(temp)
    
    stats = {
        "greedy_correct": correct,
        "greedy_total": total,
        "greedy_accuracy": correct/total,
        "greedy_none_ans": none_ans
    }
    print(stats)
    print("Done computing accuracy of greedy...")
    return stats


def greedy_eval(args, gen_config):
    set_seeds(args.seed)

    rank = int(os.environ.get("RANK"))
    model, tokenizer = create_model(args.policy_model_name, rank)
    model.eval()

    eval_ds, data_collator = create_datasets(args, tokenizer, args.data_path)
    dataloader = DataLoader(
            eval_ds,
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=args.batch_size * 4,
            # sampler=sampler,
            drop_last=False,
        )

    stop_tokens = ["Problem:"]
    acc_stats = _compute_acc(dataloader, 
                            model, 
                            tokenizer, 
                            gen_config, 
                            stop_tokens, 
                            eval_ds.prompt.ans_delimiter)
    return acc_stats


if __name__ == "__main__":
    greedy_eval()
