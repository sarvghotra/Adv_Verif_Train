# Read dataset (using datasets lib) from disk
# First, count the number of unique queries (in 'text' column)
#   - For a query, only use the first 7 words
# Then, create a train-val split (95-5)
#  - For each query (first 7 words), randomly assign it to train or val
# Save the 'train' and 'val' datasets object to disk


from datasets import load_from_disk, DatasetDict
import random
from collections import defaultdict

def create_train_val_split_math_shepered_orm_data(input_path, output_path):
    # Load the dataset from the input path
    dataset = load_from_disk(input_path)

    # Create a dictionary to group rows by the first 7 words in the 'text' column
    query_groups = defaultdict(list)
    for idx, row in enumerate(dataset):
        query = " ".join(row['text'].split()[:7])
        query_groups[query].append(idx)

    # Print the number of unique queries
    print(f"Number of unique queries: {len(query_groups)}")

    # Randomly assign each unique query to train or val
    train_indices, val_indices = [], []
    for query, indices in query_groups.items():
        if random.random() < 0.95:
            train_indices.extend(indices)
        else:
            val_indices.extend(indices)

    # Print the number of queries in train and val splits
    train_queries = set(query for idx in train_indices for query in [" ".join(dataset[idx]['text'].split()[:7])])
    val_queries = set(query for idx in val_indices for query in [" ".join(dataset[idx]['text'].split()[:7])])
    print(f"Number of queries in train split: {len(train_queries)}")
    print(f"Number of queries in val split: {len(val_queries)}")

    # Split the dataset using the selected indices
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    # Create a DatasetDict object to save train and val splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'eval': val_dataset
    })

    # Save the split dataset to the output path
    dataset_dict.save_to_disk(output_path)

if __name__ == "__main__":
    seed = 102239

    # input_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset"
    # output_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/math-shepherd_math-minos_ORM_format_dataset_splits"

    input_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/pv_methd/tmp_preproc"
    output_path = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/gen/pv_methd/tmp_preproc_dataset_splits"

    create_train_val_split_math_shepered_orm_data(input_path, output_path)
    print("Done creating train-val split")
    