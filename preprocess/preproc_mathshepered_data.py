# Read Math-shepherd data
#   Schema: 
#      - input: String
#      - label: String
#      - task: String
# Filter in 'task' column = "GSM8K"
# New 'input' = Remove 'kn' token from 'input' column
# Compute label column as last_token of 'label' column = "+" -> 1, "-" -> 0
# Output schema: 'input', 'label'
# Print stats of the labels
# Print 5 random samples

# Write a python code to do the above task

import pandas as pd
import random
import json
from datasets import Dataset


def convert_mathshepherd_to_mathminos_format(input_file, output_file):
    # Read the input data from jsonl
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    
    ############################
    ######### Temp code ########
    ############################
    
    # filtered_df = df[df['task'] == "GSM8K"].copy()

    # filtered_df['new_input'] = filtered_df['input'].str.replace('ки', '', regex=False)
    # filtered_df['new_label'] = filtered_df['label'].apply(lambda x: 1 if x.strip()[-1] == '+' else 0)
    
    # # print 5 random samples
    # random_samples = filtered_df.sample(5)
    # print("\n5 Random Samples:")
    # print(random_samples['input'].iloc[0])
    # print()
    # print(random_samples['new_input'].iloc[0])
    # print("--------------------")
    # print(random_samples['label'].iloc[0])
    # print()
    # print(random_samples['new_label'].iloc[0])

    # import sys
    # sys.exit()

    ############################
    ############################
    ############################


    # Filter rows where 'task' column equals "GSM8K"
    filtered_df = df[df['task'] == "GSM8K"].copy()
    
    # Remove 'kn' token from 'input' column
    filtered_df['text'] = filtered_df['input'].str.replace('ки', '', regex=False)
    
    # Compute 'label' as per the last token logic
    filtered_df['label'] = filtered_df['label'].apply(lambda x: 1 if x.strip()[-1] == '+' else 0)
    
    # Select only 'input' and 'label' columns
    output_df = filtered_df[['text', 'label']]
    
    # Save to output file in jsonl format
    with open(output_file, 'w') as f:
        for record in output_df.to_dict(orient='records'):
            f.write(json.dumps(record) + '\n')

    # Save to datasets library format
    dataset = Dataset.from_pandas(output_df)
    dataset.save_to_disk(output_file.replace('.jsonl', '_dataset'))
    
    # Print statistics of labels
    label_counts = output_df['label'].value_counts()
    print("Label statistics:")
    print(label_counts)
    
    # Print 5 random samples
    random_samples = output_df.sample(5)
    print("\n5 Random Samples:")
    print(random_samples)


if __name__ == '__main__':
    input_file = '/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd.jsonl'
    output_file = '/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format.jsonl'
    convert_mathshepherd_to_mathminos_format(input_file, output_file)
