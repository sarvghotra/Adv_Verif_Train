# Read data in mathshepered format
# - format: "text" "label"
# Extract query (string before " Step 1:") from "text"
# Extract final answer (string after "The answer is: {answer} ") from "text"
# Save in jsonl file
# - format: "text" "answer"

import json

# Define input and output file paths
input_file = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/tmp.jsonl"  # Path to the mathshepered format data
output_file = "/lustre/fsw/portfolios/llmservice/users/sghotra/data/math_shepherd/tmp_gsm8k_format.jsonl"  # Path to save the jsonl file

# Define the function to extract required information
def extract_data(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            # Parse the line as JSON
            entry = json.loads(line.strip())
            
            
            # Extract "text" field from the JSON entry
            text = entry.get("text", "")
            
            # Extract the query and final answer from "text"
            query_part = text.split(" Step 1:")[0]
            answer_part = int(text.split("The answer is: ")[-1][:-1])
            
            # Create a dictionary for JSONL format
            data = {
                "text": query_part,
                "answer": answer_part
            }
            
            # Write the JSONL line to the output file
            outfile.write(json.dumps(data) + "\n")

# Execute the function to process data
extract_data(input_file, output_file)
print("Done!")
