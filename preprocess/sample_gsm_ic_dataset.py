import random
import json


def random_sample_subset(file_path, num_samples, output_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    random.shuffle(data)
    data = data[:num_samples]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    random.seed(4291)

    file_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/gsm-ic/GSM-IC_mstep.json"
    num_samples = 40
    output_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/grade-school-math/gsm-ic/GSM-IC_mstep_rand40.json"

    random_sample_subset(file_path, num_samples, output_path)
