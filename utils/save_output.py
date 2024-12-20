import json
import os

from utils.extract_gsm_ans import extract_answer_from_response, remove_prefix_until


class OutputWriter:
    def __init__(self, return_seq_num, delimiter, output_file_path):
        self.delimiter = delimiter
        self.return_seq_num = return_seq_num
        self.output_path = output_file_path
        self.all_outputs = []
    
    def is_empty(self):
        if len(self.all_outputs) == 0:
            return True
        
        return False

    def write(self, step):
        print(f"written output at step: {step}")
        with open(self.output_path, 'a+') as f:
            for item in self.all_outputs:
                f.write(json.dumps(item) + '\n')
    
        self.all_outputs = []

    def collect(self, inputs_string, outputs_string, targets, ids, questions):
        for idx in range(len(inputs_string)):
            gt_ans = int(targets[idx].cpu().numpy())
            question = questions[idx]
            question = remove_prefix_until([question], until='Problem:')[0]
            
            for i in range(self.return_seq_num):
                response = outputs_string[self.return_seq_num*idx+i].replace(inputs_string[idx], '')
                # print("=========================")
                # print(outputs_string[self.return_seq_num*idx+i])
                # print("++++++++++++")
                # print(response)
                # print("=========================")
                
                response = response.rstrip()
                final_ans = extract_answer_from_response(response, delimiter='The final answer is ')
                curr_correct = 1 if final_ans == gt_ans else 0

                
                response = remove_prefix_until([response], until='assistant')[0]
                response = remove_prefix_until([response], until='Solution:\n')[0]
                temp = {
                    # "input": inputs_string[idx],
                    "question": question,
                    "target": gt_ans,
                    "response": response,
                    "label": curr_correct,
                    "id": int(ids[idx]),
                    "gen_id": i
                }
                self.all_outputs.append(temp)


def load_generated_samples(output_dir):
    all_outputs = []
    # read jsonl files
    for file in os.listdir(output_dir):
        with open(os.path.join(output_dir, file), 'r') as f:
            for line in f:
                all_outputs.append(json.loads(line))
    return all_outputs
