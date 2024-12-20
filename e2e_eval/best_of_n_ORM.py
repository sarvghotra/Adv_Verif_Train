import argparse
import json
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
from utils.set_seeds import set_seeds
from utils.create_model import create_model
from utils.utils import create_datasets
from utils.extract_gsm_ans import extract_answer_from_response
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GenerationConfig
from utils.generate_samples import SampleLLMPolicy
from utils.save_output import OutputWriter, load_generated_samples
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets_classes.prompts import prepare_mistral_input
from torch.nn.parallel import DistributedDataParallel as DDP


# generate samples
def generate_samples(args, gen_config):
    set_seeds(args.seed)
    return_seq_num = args.return_seq_num

    local_rank = int(os.environ.get("LOCAL_RANK"))
    model, tokenizer = create_model(args.policy_model_name, local_rank)
    model.eval()

    eval_ds, data_collator = create_datasets(args, tokenizer, args.data_path)
    dataloader = DataLoader(
            eval_ds,
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=args.batch_size,
            sampler=DistributedSampler(eval_ds),
            drop_last=False,
        )

    # stop_tokens = ["Problem:"]
    # generation_config = GenerationConfig(
    #     temperature=1.0,
    #     do_sample=False,
    #     max_new_tokens=256,
    #     # top_p=1,
    #     # temperature=0.8 if args.diverse_beam > 1 else 1.0,
    #     # num_beam_groups=args.diverse_beam,
    #     # diversity_penalty=1.0,
    #     # num_beams=return_seq_num,
    # )

    sample_generator = SampleLLMPolicy(model, 
                                       tokenizer, 
                                       **gen_config)

    output_dir = os.path.join(args.out_path, f"tmp_{args.out_file}")
    local_rank = int(os.environ.get("LOCAL_RANK"))
    global_rank = int(os.environ.get("RANK"))
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # remove all files in the output_dir
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    dist.barrier()
    
    output_file_path = os.path.join(output_dir, f"output_{global_rank}_{local_rank}.jsonl")
    output_collector = OutputWriter(return_seq_num, eval_ds.prompt.ans_delimiter, output_file_path=output_file_path)
    dt_prompt_tokens_len = eval_ds.txt_prompt_len

    print("Generating samples...")
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        inputs_string, outputs_string = sample_generator.sample(batch)
        targets = batch['targets']
        ids = batch['id'].cpu().numpy()

        questions = [inputs_string[i][dt_prompt_tokens_len:] for i in range(len(inputs_string))]
        output_collector.collect(inputs_string, outputs_string, targets, ids, questions)

        # if step > 0 and step % 25 == 0:
        #     output_collector.write(step)
    
    # if not output_collector.is_empty():
    #     output_collector.write(step)
    print("Done generating samples...")

    # group by id
    grouped_outputs = {}
    for output in output_collector.all_outputs:
        if output['id'] not in grouped_outputs:
            output['response'] = [output['response']]
            grouped_outputs[output['id']] = output
        else:
            grouped_outputs[output['id']]['response'].append(output['response'])
        
    gen_outputs = list(grouped_outputs.values())
    return gen_outputs, eval_ds.prompt.ans_delimiter


# run orm for each q-r (DS x N) pairs
def add_orm_score(gen_samples, args):
    model_name_or_path = args.reward_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        # num_labels=1,
        torch_dtype=torch.bfloat16,
    )
    
    device = torch.cuda.current_device()
    reward_model.to(device)
    reward_model = DDP(reward_model, device_ids=[device])
    reward_model.eval()

    print("Adding ORM score...")

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for d in tqdm(gen_samples, total=len(gen_samples)):
            ques_inst = d['question']
            respones = d['response']
            # gt_ans = d['target']
            
            input_data_batch = prepare_mistral_input(instruction=ques_inst, responses=respones, tokenizer=tokenizer)  # 这是一个虚构的函数
            input_data_batch = [input_data.to(device) for input_data in input_data_batch]
            
            rewards = []
            with torch.no_grad():
                for inputs in input_data_batch:
                    reward = reward_model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        return_dict=True,)["logits"]
                    rewards.append(reward)
            
            # 将reward接输出成scalar值。如果reward本身就是一个scalar，这一步可以省略。
            rewards = [F.sigmoid(input=torch.FloatTensor([reward.item()])).item() for reward in rewards]
            # 将原始数据和计算出的reward保存到列表
            d["rewards"] = rewards
    
    print("Saving ORM score...")
    output_dir = os.path.join(args.out_path, f"tmp_{args.out_file}")
    local_rank = int(os.environ.get("LOCAL_RANK"))
    global_rank = int(os.environ.get("RANK"))
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # remove all files in the output_dir
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    dist.barrier()
    
    print("Writing ORM score...")
    output_file_path = os.path.join(output_dir, f"output_{global_rank}_{local_rank}.jsonl")
    with open(output_file_path, 'w') as f:
        for d in gen_samples:
            f.write(json.dumps(d) + '\n')
    
    print("Done writing ORM score...")
    # gen_samples contains reward scores too
    return output_dir


def compute_accuracy(gen_sample_w_orm_score, ans_delimiter):
    total = 0
    correct = 0
    none_cnt = 0

    for data_pt in gen_sample_w_orm_score:
        # input_text = data_pt["input"]
        responses = data_pt["response"]
        rewards = data_pt["rewards"]
        target = data_pt["target"]

        # Extract answers and select the best based on the reward
        best_answer = None
        best_reward = float('-inf')
        for i, response in enumerate(responses):
            answer = extract_answer_from_response(response, delimiter=ans_delimiter)
            if answer is not None and rewards[i] > best_reward:
                best_reward = rewards[i]
                best_answer = answer
        
        # Compare the best answer with the target
        total += 1
        if best_answer == target:
            correct += 1
        
        if best_answer is None:
            none_cnt += 1
        
    # Calculate and print accuracy
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    none_frac = none_cnt / total
    print(f"None answers percentage: {none_frac * 100:.2f}% ({none_cnt}/{total})")
    metric_stats = {"bon_orm_accuracy": accuracy * 100, 
                    "bon_orm_correct": correct, 
                    "bon_orm_total": total, 
                    "bon_orm_none_percentage": none_frac * 100, 
                    "bon_orm_none_ans_count": none_cnt}
    return metric_stats


def best_of_n_ORM_eval(args, gen_config):
    metric_stats = None
    gen_samples, ans_delimiter = generate_samples(args, gen_config)
    gen_sample_w_orm_score_outdir = add_orm_score(gen_samples, args)

    # # FIXME: remove this
    # gen_sample_w_orm_score_outdir = "/lustre/fsw/portfolios/llmservice/users/sghotra/models/adv_verif/bl/evals/tmp_itr1_stg1_base-inst_mistral_1e-5/"
    # ans_delimiter = "The final answer is "

    global_rank = int(os.environ.get("RANK"))
    if global_rank == 0:
        gen_sample_w_orm_score = load_generated_samples(gen_sample_w_orm_score_outdir)
        metric_stats = compute_accuracy(gen_sample_w_orm_score, ans_delimiter)
        print("Done computing accuracy of best of n ORM...")
    
    dist.barrier()
    return metric_stats


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args = args.parse_args()
    best_of_n_ORM_eval(args)