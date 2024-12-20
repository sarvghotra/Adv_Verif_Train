import random
import os
import json

from utils.extract_gsm_ans import extract_answer_from_response
from fuzzywuzzy import fuzz
from datasets import Dataset
from datasets import load_from_disk
from datasets import DatasetDict


def stats(joined_data):
    # compute the stats
    per_prompt_label_frac = {}

    for line in joined_data:
        id = line['id']
        label = line['label']
        if id not in per_prompt_label_frac:
            per_prompt_label_frac[id] = []
            
        per_prompt_label_frac[id].append(label)

    pos_labels = 0
    total = 0
    for id in per_prompt_label_frac:
        pos_labels += sum(per_prompt_label_frac[id])
        total += len(per_prompt_label_frac[id])
        pos_label_frac = sum(per_prompt_label_frac[id]) / len(per_prompt_label_frac[id])
        per_prompt_label_frac[id] = [pos_label_frac, 1 - pos_label_frac]

    # overall frac
    print("Overall frac: ", round(pos_labels / total, 2), round(1 - (pos_labels / total), 2))

    # per prompt label frac histogram

    import matplotlib.pyplot as plt
    import numpy as np

    pos_label_fracs = [per_prompt_label_frac[id][0] for id in per_prompt_label_frac]
    neg_label_fracs = [per_prompt_label_frac[id][1] for id in per_prompt_label_frac]

    # labels fracs are b/w 0 and 1
    plt.hist(pos_label_fracs, bins=20, alpha=0.5, label='pos label fracs')
    plt.xlabel('Per prompt positive label frac')
    plt.ylabel('Frequency')
    plt.title('Gen data label frac histogram')

    # TODO: save it


def print_rand_samples(data):
    list_grouped_data = list(data.values())
    random.shuffle(list_grouped_data)
    rand_10 = list_grouped_data[:5]

    print("-----------")
    print("Len: ", len(rand_10))
    print("-----------")

    for grp in rand_10:
        for res, idx in zip(grp, range(3)):
            print(res[0])
            print("Target: ", res[1])
            print()
        print("************")


def remove_pred_ans_trail(response, pred_ans):
    response = response.split("The final answer is")[0] + "The final answer is " + str(pred_ans)
    return response


def depulicate_stat(rand_samples):
    # Function to perform fuzzy matching between two strings
    def fuzzy_match(string1, string2):
        # Compute the similarity ratio between the two strings
        ratio = fuzz.ratio(string1, string2)
        return ratio

    sim_count = 0
    for group in rand_samples:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                similarity_ratio = fuzzy_match(group[i], group[j])
                # print(f"Similarity ratio: {similarity_ratio}%")
                if similarity_ratio > 80:
                    # print("Similarity ratio: ", similarity_ratio)
                    # print(group[i])
                    # print(group[j])
                    # print("--------")
                    sim_count += 1
    print("=====================================")
    print("Total similar pairs: ", sim_count)


def save_dataset(output_dir, out_data):
    data_dict = {key: [d[key] for d in out_data] for key in out_data[0]}
    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk(output_dir)


##################### Text example format ########################
# A store is having an anniversary sale. A tub of ice cream costs $2 less than its original price of $12 and the juice is sold at $2 for 5 cans. How much will you pay if you buy two tubs of ice cream and 10 cans of juice?
# Step 1: Percy swims 2 hours (1 hour in the morning and 1 in the afternoon) each school day. He swims this for 5 days a week. This is 2 * 5 = 10 hours per week.
# Step 2: This makes 10 hours each week from school. He also swims 3 hours on the weekend. So all together, 10 + 3 = 13 hours per week.
# Step 3: There are 4 weeks. So 13 * 4 is 52.
# The final answer is 52
##################################################################
def post_process(data_file, gen_data_dir, output_dir, delimiter, has_flawed_sol):
    # "The final answer is "
    data = {}
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            line = json.loads(line.strip())
            line['id'] = i
            data[i] = line
    
    gen_data = []

    # m = [x for x in os.listdir(gen_data_dir)]
    # print("gen_data_dir: ", gen_data_dir)
    # print("os.listdir(gen_data_dir): ", m)

    all_ids = []

    for gen_data_file in os.listdir(gen_data_dir):
        gen_data_file = os.path.join(gen_data_dir, gen_data_file)
        
        # print("gen_data_file: ", gen_data_file)
        
        with open(gen_data_file, 'r') as f:
            for i, line in enumerate(f):
                # print(i)
                line = json.loads(line.strip())
                gen_data.append(line)
                all_ids.append(line["id"])
        
        # print()
    
    print("Total number of examples in input: ", len(data))
    print("Total number of examples in gen: ", len(gen_data))
    all_ids = set(all_ids)
    print("Total number of ids in gen: ", len(all_ids))

    filtered_gen_data = []
    all_ids = []

    nb_non_int_ans = 0
    nb_solution_word_not_found = 0

    for i in range(len(gen_data)):
        response = gen_data[i]['response']

        # FIXME: remove it
        if has_flawed_sol:
            response = response.split("Flawed ")[1] if "Flawed " in response else ""
        
        response = response.rstrip()
        pred_ans = extract_answer_from_response(response, delimiter=delimiter)
        if not isinstance(pred_ans, int):
            nb_non_int_ans += 1
            continue

        response = remove_pred_ans_trail(response, pred_ans)
        # remove "Solution:\n"
        try:
            response = response.split("Solution:\n")[1]
        except:
            nb_solution_word_not_found += 1
            print("Error in response: ", response)
            continue
        
        gen_data[i]['response'] = response
        all_ids.append(gen_data[i]["id"])

        # print("++++++++++++++")
        # print(response)
        # print("++++++++++++")

        filtered_gen_data.append(gen_data[i])

    print("Original gen_data length: ", len(gen_data))
    print("Filtered gen_data length: ", len(filtered_gen_data))
    all_ids = set(all_ids)
    print("Total number of ids in filtered gen: ", len(all_ids))

    print("Number of non-int answers: ", nb_non_int_ans)
    print("Number of solution word not found: ", nb_solution_word_not_found)

    gen_data = filtered_gen_data

    # join based on id
    joined_data = []

    for line in gen_data:
        id = line['id']
        orig_line = data[id]
        joined_line = {**orig_line, **line}
        joined_data.append(joined_line)

    # ORM training format
    # text (query: from orig data + response: from gen data) and label

    out_data = []
    for line in joined_data:
        query = line['question']
        response = line['response']
        text = query + "\n" + response
        out = {"text": text, **line}

        out_data.append(out)

    # group the data based on ID and random sample 10 groups

    grouped_data = {}
    for line in out_data:
        id = line['id']
        if id not in grouped_data:
            grouped_data[id] = []
        # query = line['question']
        # response = line['response']
        text = line['text']
        target = line['target']
        grouped_data[id].append((text, target))

    print_rand_samples(grouped_data)

    # output_dir = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/pv_methd/preproc_gsm8k_neg/full"
    print(f"Saving in {output_dir}")
    save_dataset(output_dir, out_data)


# For positive-only data (baseline)
def train_val_split(args):
    random.seed(args.seed)
    input_dir = args.out_path + '/datasets/pos'
    output_save_dir = os.path.join(args.out_path, 'datasets_pos_splits')

    data = load_from_disk(input_dir)

    # data_grp = {}
    # for i in range(len(data)):
    #     id = data[i]["id"]
        
    #     if id in data_grp:
    #         data_grp[id].append(data[i])
    #     else:
    #         data_grp[id] = [data[i]]

    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()

    all_ids = []
    for d in data:
        all_ids.append(d["id"])
    
    all_ids = set(all_ids)
    print("=========================")
    print("Total ids: ", len(all_ids))
    val_ids = random.sample(all_ids, 500)
    print("Selecting 500 ids for val set")
    print("=========================")
    
    train_data = []
    val_data = []

    for d in data:
        id = d["id"]
        if id in val_ids:
            val_data.append(d)
        else:
            train_data.append(d)

    print("Train data: ", len(train_data))
    print("Val data: ", len(val_data))

    train_data_dict = {key: [d[key] for d in train_data] for key in train_data[0]}
    train_dataset = Dataset.from_dict(train_data_dict)

    val_data_dict = {key: [d[key] for d in val_data] for key in val_data[0]}
    val_dataset = Dataset.from_dict(val_data_dict)

    dataset_dict = DatasetDict({
            'train': train_dataset,
            'eval': val_dataset
        })

    os.mkdir(output_save_dir)

    print("Saved the data in: ", output_save_dir)
    dataset_dict.save_to_disk(output_save_dir)
