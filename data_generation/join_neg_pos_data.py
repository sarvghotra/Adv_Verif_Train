import os
import random
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk


def join_neg_pos_policies_data(out_paths, save_dir):
    # read the datasets (huggingface)

    print("out_paths: ", out_paths)

    pos_data_path = out_paths[0]
    neg_data_path = out_paths[1]

    assert "pos" in pos_data_path
    assert "neg" in neg_data_path

    pos_data = load_from_disk(pos_data_path)
    neg_data = load_from_disk(neg_data_path)
    
    print(pos_data[0].keys())
    print("Pos data: ", len(pos_data))

    # from neg dataset, remove responses w/ label=1
    print("Before filtering: ", len(neg_data))
    neg_data = neg_data.filter(lambda x: x["label"] == 0)
    print("After filtering: ", len(neg_data))

    # group by "ids"
    pos_data_grp = {}
    for i in range(len(pos_data)):
        id = pos_data[i]["id"]
        
        if id in pos_data_grp:
            pos_data_grp[id].append(pos_data[i])
        else:
            pos_data_grp[id] = [pos_data[i]]

    neg_data_grp = {}
    for i in range(len(neg_data)):
        
        id = neg_data[i]["id"]
        if id in neg_data_grp:
            neg_data_grp[id].append(neg_data[i])
        else:
            neg_data_grp[id] = [neg_data[i]]


    

    # merge the pos and neg datasets: 50% of responses from each dataset.

    incomplete_dpoints = 0
    incomplete_dpoints_neg = 0
    incomplete_dpoints_pos = 0

    merged_data = []
    all_ids = set(pos_data_grp.keys()) | set(neg_data_grp.keys())
    for id in all_ids:
        pos_dp_h = []
        neg_dp_h = []

        if id in neg_data_grp:
            neg_dp = neg_data_grp[id]
            # print(neg_dp)
            # print(len(neg_dp))
            random.shuffle(neg_dp)
            neg_dp_h = neg_dp[:8]

            for dp in neg_dp_h:
                dp["policy"] = "neg"
        
        
        if id in pos_data_grp:
            pos_dp = pos_data_grp[id]
            random.shuffle(pos_dp)
            # FIXME: remove it
            n = max(8, 16 - len(neg_dp_h))
            pos_dp_h = pos_dp[:n]

            for dp in pos_dp_h:
                dp["policy"] = "pos"
        

        # if len(neg_dp_h) < 8:
        #     neg_dp_h = neg_dp.shuffle()[:8]
        #     if len(neg_dp_h) < 8:
        #         pos_dp_h = pos_dp.shuffle()[:8]
        
        # if len(pos_dp_h) < 8:
        #     pos_dp_h = pos_dp.shuffle()[:8]

        dp = pos_dp_h + neg_dp_h
        merged_data.extend(dp)

        if len(dp) < 16:
            incomplete_dpoints += 1
        
    print(f"Number of incomplete datapoints: {incomplete_dpoints}")

    print(len(all_ids))
    print("Merged data: ", len(merged_data))
    print(merged_data[0].keys())






    # print a random some sample
    rand_i = list(range(len(merged_data)))
    random.shuffle(rand_i)
    rand_i = rand_i[:5]

    # for i in rand_i:
    #     print(merged_data[i]['policy'])
    #     print(merged_data[i]['label'])
    #     print()
    #     print(merged_data[i]['text'])
    #     print("===================================")







    # random select 500 ids for val set
    random.seed(6730)
    all_ids = list(all_ids)
    all_ids.sort()
    val_ids = random.sample(all_ids, 500)

    train_data = []
    val_data = []

    for dp in merged_data:
        if dp["id"] in val_ids:
            val_data.append(dp)
        else:
            train_data.append(dp)

    print("Train data: ", len(train_data))
    print("Val data: ", len(val_data))

    # Now, split val data based on policy into pos_val and neg_val
    pos_val_data = []
    neg_val_data = []

    for dp in val_data:
        if dp["policy"] == "pos":
            pos_val_data.append(dp)
        else:
            neg_val_data.append(dp)

    print("Pos val data: ", len(pos_val_data))
    print("Neg val data: ", len(neg_val_data))






    

    train_data_dict = {key: [d[key] for d in train_data] for key in train_data[0]}
    train_dataset = Dataset.from_dict(train_data_dict)

    val_pos_data_dict = {key: [d[key] for d in pos_val_data] for key in pos_val_data[0]}
    val_pos_dataset = Dataset.from_dict(val_pos_data_dict)

    val_neg_data_dict = {key: [d[key] for d in neg_val_data] for key in neg_val_data[0]}
    val_neg_dataset = Dataset.from_dict(val_neg_data_dict)

    output_dir = os.path.join(save_dir, "datasets_bse-inst_pos_neg_splits")
    data = DatasetDict({"train": train_dataset, "val_pos": val_pos_dataset, "val_neg": val_neg_dataset})

    data.save_to_disk(output_dir)

    print("Saved train val_pos and val_neg splits in: ", output_dir)
