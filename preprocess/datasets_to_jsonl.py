import os
import datasets
import json

def datasets_to_jsonl(input_dir, output_file):
    # Load the dataset from the directory
    dataset = datasets.load_from_disk(input_dir)
    
    # Extract the 'eval' split from the dataset
    eval_split = dataset['eval']
    
    # Write the 'eval' split to a JSONL file
    with open(output_file, 'w') as f:
        for record in eval_split:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    input_dir = '/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits'
    output_file = '/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits/eval.jsonl'
    
    datasets_to_jsonl(input_dir, output_file)
    print("Done!")
    