import argparse
import json
import os
import matplotlib.pyplot as plt
from transformers import GenerationConfig
from data_generation.gen_gsm_data import ddp_setup
from greedy_eval import greedy_eval
from best_of_n_ORM import best_of_n_ORM_eval
# from self_const_eval import self_const_eval
# from best_of_n_maj import best_of_n_maj_eval


def write_results(args, res_dict):
    output_file = os.path.join(args.out_path, f"{args.out_file}.json")
    with open(output_file, 'w') as f:
        json.dump(res_dict, f)


def plot_n_save_results(args, res_dict):
    # draw bar chart
    # keep only "_accuracy" keys
    res_dict = {k: v for k, v in res_dict.items() if k.endswith("_accuracy")}
    plt.bar(res_dict.keys(), res_dict.values())
    out_path = os.path.join(args.out_path, f"{args.out_file}.png")
    plt.savefig(out_path)


def is_main_process():
    return int(os.environ.get("RANK")) == 0


def run_all_evals(args):
    local_rank = int(os.environ.get("LOCAL_RANK"))
    ddp_setup(local_rank)

    g_res_dict = None
    if args.run_greedy:
        greedy_gen_config = GenerationConfig(
            temperature=1.0,
            do_sample=False,
            max_new_tokens=256,
            # top_p=1,
            # temperature=0.8 if args.diverse_beam > 1 else 1.0,
            # num_beam_groups=args.diverse_beam,
            # diversity_penalty=1.0,
            # num_beams=return_seq_num,
        )
        g_res_dict = greedy_eval(args, greedy_gen_config)
    
    
    # BOS_maj_res_dict = best_of_n_maj_eval(args)
    gen_config = {
        "do_sample": True, 
        "temperature": 1.0, 
        "max_new_tokens": 256,
        "return_seq_num": args.return_seq_num
    }
    BOS_ORM_res_dict = best_of_n_ORM_eval(args, gen_config)
    # sc_res_dict = self_const_eval(args)

    if is_main_process():
        all_res = {}
        if g_res_dict is not None:
            all_res = g_res_dict
        
        all_res.update(BOS_ORM_res_dict)

        write_results(args, all_res)
        plot_n_save_results(args, all_res)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--policy_model_name", type=str)
    # parser.add_argument("--rank", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out_path", type=str)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--run_greedy", action="store_true")

    parser.add_argument("--reward_model_name", type=str, default="")
    parser.add_argument("--return_seq_num", type=int, default=1)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_evals(args)
