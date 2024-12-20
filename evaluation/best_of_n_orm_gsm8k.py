# Author: Sarvjeet Singh Ghotra

import os
import sys
# import re
import json
from utils.extract_gsm_ans import extract_answer_from_response
# from fractions import Fraction

# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         pass
#     try:
#         import unicodedata
#         unicodedata.numeric(s)
#         return True
#     except (TypeError, ValueError):
#         pass
#     return False

# def extract_answer_from_response(completion):
#     text = completion.split('The answer is: ')
#     if len(text) > 1:
#         extract_ans = text[-1].strip()
#         match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
#         if match:
#             if '/' in match.group():
#                 denominator = match.group().split('/')[1]
#                 numerator = match.group().split('/')[0]
#                 if is_number(denominator) and is_number(numerator):
#                     if denominator == '0':
#                         return round(float(numerator.replace(',', '')))
#                     else:
#                         frac = Fraction(match.group().replace(',', ''))
#                         num_numerator = frac.numerator
#                         num_denominator = frac.denominator
#                         return round(float(num_numerator / num_denominator))
#                 else:
#                     return None
#             else:
#                 if float(match.group().replace(',', '')) == float('inf'):
#                     return None
#                 return round(float(match.group().replace(',', '')))
#         else:
#             return None
#     else:
#         return None

def best_of_n_orm_result(res_dir, ans_delimiter):
    total = 0
    correct = 0

    # Iterate through all files in the provided directory
    for filename in os.listdir(res_dir):
        file_path = os.path.join(res_dir, filename)
        # if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                # try:
                data = json.loads(line)
                input_text = data["input"]
                responses = data["response"]
                rewards = data["rewards"]
                target = data["target"]

                # Extract answers and select the best based on the reward
                best_answer = None
                best_reward = float('-inf')
                for i, response in enumerate(responses):
                    # print(response)
                    answer = extract_answer_from_response(response, delimiter=ans_delimiter)
                    # print(answer)
                    # print("--")
                    if answer is not None and rewards[i] > best_reward:
                        best_reward = rewards[i]
                        best_answer = answer

                # print(f"Best answer: {best_answer}, Target: {target}")
                # Compare the best answer with the target
                total += 1
                if best_answer is not None and best_answer == target:
                    correct += 1
                # print(f"Total: {total}, Correct: {correct}")
                # except json.JSONDecodeError:
                #     print(f"Error decoding JSON from line in file {filename}")
                # except KeyError as e:
                #     print(f"Missing key {e} in file {filename}")

    # Calculate and print accuracy
    # if total > 0:
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    # else:
    #     print("No valid data found for evaluation.")


if __name__ == '__main__':
    res_dir = sys.argv[1]
    ans_delimiter = sys.argv[2]
    best_of_n_orm_result(res_dir, ans_delimiter)
