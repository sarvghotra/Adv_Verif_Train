import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils.set_seeds import set_seeds
from utils.create_model import create_model
from utils.utils import create_datasets
from utils.extract_gsm_ans import extract_answer_from_response
from torch.utils.data import DataLoader
from transformers import GenerationConfig
from utils.generate_samples import SampleLLMPolicy
from utils.save_output import OutputWriter
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets_classes.prompts import prepare_mistral_input
from collections import Counter


# generate samples
def generate_samples(args):
    set_seeds(args.seed)

    base_model = args.model_path
    data_path = args.data_path
    batch_size = args.batch_size
    return_seq_num = args.return_seq_num

    set_seeds(args.seed)

    model, tokenizer = create_model(args.model_name, args.rank)
    model.eval()

    eval_ds, data_collator = create_datasets(args, tokenizer, args.data_path)
    dataloader = DataLoader(
            args.eval_dataset, 
            shuffle=False, 
            collate_fn=data_collator, 
            batch_size=args.batch_size,
            # sampler=sampler,
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

    sample_generator = SampleLLMPolicy(model, tokenizer, do_sample=True, temperature=1.0, max_new_tokens=256, return_seq_num=return_seq_num)

    output_dir = args.out_path
    os.makedirs(args.out_path, exist_ok=True)
    output_collector = OutputWriter(return_seq_num, eval_ds.prompt.ans_delimiter, output_file_path=None)

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

            inputs_string, outputs_string  = sample_generator.sample(batch)
            targets = batch['targets']
            ids = batch['id'].cpu().numpy()
            
            output_collector.collect(inputs_string, outputs_string, targets, ids)

            # if step > 0 and step % 25 == 0:
            #     writer.write(step)
    
    return output_collector.all_outputs, eval_ds.prompt.ans_delimiter



# run orm for each q-r (DS x N) pairs
def select_maj_resp(gen_samples, args, ans_delimiter):
    
    for d in gen_samples:
        ques_inst = d['question']
        respones = d['response']
        gt_ans = d['target']
        
        final_ans = [extract_answer_from_response(resp, delimiter=ans_delimiter) for resp in respones]
        maj_ans_resp_idx = Counter(final_ans).most_common(1)[0][0]
        maj_resp = respones[maj_ans_resp_idx]
        d["response"] = maj_resp
    
    # gen_samples contains reward scores too
    return gen_samples


def compute_accuracy(gen_sample_w_orm_score, ans_delimiter):
    total = 0
    correct = 0
    none_cnt = 0

    for data_pt in gen_sample_w_orm_score:
        input_text = data_pt["input"]
        response = data_pt["response"]
        target = data_pt["target"]

        answer = extract_answer_from_response(response, delimiter=ans_delimiter)

        # Compare the best answer with the target
        total += 1
        if answer == target:
            correct += 1
        
        if answer is None:
            none_cnt += 1
        
    # Calculate and print accuracy
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    none_frac = none_cnt / total
    print(f"None answers percentage: {none_frac * 100:.2f}% ({none_cnt}/{total})")
    metric_stats = {"accuracy": accuracy * 100, 
                    "correct": correct, 
                    "total": total, 
                    "none_percentage": none_frac * 100, 
                    "none_ans_count": none_cnt}
    return metric_stats


def best_of_maj(args):
    gen_samples, ans_delimiter = generate_samples(args)
    maj_resp_samples = select_maj_resp(gen_samples, args, ans_delimiter)
    metric_stats = compute_accuracy(maj_resp_samples, ans_delimiter)
    return metric_stats


if __name__ == "__main__":
    print("Boom")
