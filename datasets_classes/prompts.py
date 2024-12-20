import re


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{text}\n\n### Response:"
    ),
    "prompt_no_input_v2": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{text}\n\n### Response:"
    ),
    "prompt_metamath_mistral7b": (
        "Below is an instruction that describes a task. " 
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response: Let's think step by step."
    )
}


class LLAMA8BInstPrompt:
    def __init__(self):
        self.inst_e1 = """Given the following problem, reason and give a final answer to the problem. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
"""
        self.inst_e1_s = """Solution:
Step 1: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
The final answer is 6
"""

        self.inst_e2 = """Problem:
If there are 3 cars in the parking lot and 2 more cars arrive, how manycars are in the parking lot?
"""

        self.inst_e2_s = """Solution:
Step 1: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
The final answer is 5
"""

        self.inst_e3 = """Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
"""

        self.inst_e3_s = """Solution:
Step 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74.
Step 2: After eating 35, they had 74 - 35 = 39.
The final answer is 39
"""

        self.inst_e4 = """Problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
"""

        self.inst_e4_s = """Solution:
Step 1: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
The final answer is 8
"""

        self.inst_e5 = """Problem:
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
"""

        self.inst_e5_s = """Solution:
Step 1: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
The final answer is 9
"""

        self.inst_e6 = """Problem:
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
"""

        self.inst_e6_s = """Solution:
Step 1: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added.
Step 2: 9 + 20 is 29.
The final answer is 29
"""

        self.inst_e7 = """Problem:
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
"""

        self.inst_e7_s = """Solution:
Step 1: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35.
Step 2: After losing 2 more, he had 35 - 2 = 33 golf balls.
The final answer is 33
"""

        self.inst_e8 = """Problem:
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
"""

        self.inst_e8_s = """Solution:
Step 1: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
Step 2: So she has 23 - 15 dollars left. 23 - 15 is 8.
The final answer is 8
"""
        self.q_template = """Problem:
{question}
"""
        self.ans_delimiter = "The final answer is "
        self.messages = [
            {"role": "user", "content": self.inst_e1},
            {"role": "assistant", "content": self.inst_e1_s},
            {"role": "user", "content": self.inst_e2},
            {"role": "assistant", "content": self.inst_e2_s},
            {"role": "user", "content": self.inst_e3},
            {"role": "assistant", "content": self.inst_e3_s},
            {"role": "user", "content": self.inst_e4},
            {"role": "assistant", "content": self.inst_e4_s},
            {"role": "user", "content": self.inst_e5},
            {"role": "assistant", "content": self.inst_e5_s},
            {"role": "user", "content": self.inst_e6},
            {"role": "assistant", "content": self.inst_e6_s},
            {"role": "user", "content": self.inst_e7},
            {"role": "assistant", "content": self.inst_e7_s},
            {"role": "user", "content": self.inst_e8},
            {"role": "assistant", "content": self.inst_e8_s},
        ]

    def __call__(self, question):
        ques = self.q_template.format(question=question)
        mess_w_ques = self.messages + [{"role": "user", "content": ques}]
        return mess_w_ques


class LLAMA8BInstNegPrompt:
    def __init__(self):
        self.inst_e1 = """Given the following problem, reason and give a final answer to the problem. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem. Try to have only one calculation per step. As an additional education excercise for the user, edit the above solution to produce a subtly flawed reasoned solution with a wrong final answer to the problem. It's very important that the flawed solution is has similar *length* as the correct solution.
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
"""

        self.inst_e1_s = """Solution:
Step 1: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
The final answer is 6

Flawed Solution:
Step 1: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 7.
The final answer is 7

"""

        self.inst_e5 = """Problem:
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
"""

        self.inst_e5_s = """Solution:
Step 1: Shawn got 1 + 1 = 2 toys from his parents.
Step 2: Shawn started with 5 toys. Now, in total he has 5 + 2 = 7.
The final answer is 7

Flawed Solution:
Step 1: Shawn got 2 toys each from his mom and dad, 2 + 2 = 4.
Step 2: Shawn started with 5 toys and with 4 more toys he has 5 + 4 = 9.
The final answer is 9

"""


        self.inst_e3 = """Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
"""

        self.inst_e3_s = """Solution:
Step 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 35 + 42 = 77.
Step 2: After eating 35, they had 77 - 32 = 45.
The final answer is 45

Flawed Solution:
Step 1: Leah and her sister originally had a total of 32 + 42 = 74 chocolates.
Step 2: After eating 35, Leah's sister had 42 - 35 = 7.
The final answer is 7

"""

        self.inst_e6 = """Problem:
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
"""

        self.inst_e6_s = """Solution:
Step 1: There were 9 computers in the server room initially.
Step 2: 5 computers were installed each day for 4 days (Monday to Thursday). Total computers installed is 5 * 4 = 20.
Step 3: Now we have 9 (initial) + 20 (installed) = 29.
The final answer is 29

Flawed Solution:
Step 1: There were 9 computers in the server room initially.
Step 2: 5 computers were installed each day for 3 days (Monday to Thursday). Total computers installed is 5 * 3 = 15.
Step 3: Now we have 9 (initial) + 15 (installed) = 24.
The final answer is 24

"""

        self.q_template = """Problem:
{question}
"""
        self.ans_delimiter = "The final answer is "
        self.messages = [
            {"role": "user", "content": self.inst_e1},
            {"role": "assistant", "content": self.inst_e1_s},
            {"role": "user", "content": self.inst_e5},
            {"role": "assistant", "content": self.inst_e5_s},
            {"role": "user", "content": self.inst_e3},
            {"role": "assistant", "content": self.inst_e3_s},
            {"role": "user", "content": self.inst_e6},
            {"role": "assistant", "content": self.inst_e6_s}
        ]

    def __call__(self, question):
        ques = self.q_template.format(question=question)
        mess_w_ques = self.messages + [{"role": "user", "content": ques}]
        return mess_w_ques


class LLAMA8BBasePrompt:
    def __init__(self):
        self.prompt = """Given the following problem, reason and give a final answer to the problem. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"

Solution:
Step 1: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.
The final answer is 6


Problem:
If there are 3 cars in the parking lot and 2 more cars arrive, how manycars are in the parking lot?

Solution:
Step 1: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
The final answer is 5


Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Solution:
Step 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74.
Step 2: After eating 35, they had 74 - 35 = 39.
The final answer is 39


Problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

Solution:
Step 1: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
The final answer is 8


Problem:
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

Solution:
Step 1: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
The final answer is 9


Problem:
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

Solution:
Step 1: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added.
Step 2: 9 + 20 is 29.
The final answer is 29


Problem:
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Solution:
Step 1: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35.
Step 2: After losing 2 more, he had 35 - 2 = 33 golf balls.
The final answer is 33


Problem:
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

Solution:
Step 1: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
Step 2: So she has 23 - 15 dollars left. 23 - 15 is 8.
The final answer is 8


Problem:
"""
        self.ans_delimiter = "The final answer is "
    
    def __call__(self, question):
        prompt_input = self.prompt + question + "\n\nSolution:\n"
        return prompt_input


class LLAMA8BBaseNegPrompt:
    def __init__(self):
        self.prompt = """Given the following problem, produce subtly flawed reason and give a wrong final answer to the problem. Your response should end with \"The final answer is [answer]\" where [answer] is the response to the problem.
Problem:
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"

Solution:
Step 1: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 7.
The final answer is 7


Problem:
If there are 3 cars in the parking lot and 2 more cars arrive, how manycars are in the parking lot?

Solution:
Step 1: There are originally 3 cars. 2 more cars arrive. Additional car already in the lot compared to the arrived ones 3 - 2 = 1.
The final answer is 1


Problem:
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

Solution:
Step 1: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 35 + 42 = 77.
Step 2: After eating 35, they had 77 - 32 = 45.
The final answer is 45


Problem:
Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

Solution:
Step 1: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 9.
The final answer is 9


Problem:
Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

Solution:
Step 1: Shawn got 1 + 1 = 2 toys from his parents.
Step 2: Shawn started with 5 toys. Now, in total he has 5 + 2 = 7.
The final answer is 7


Problem:
There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

Solution:
Step 1: For each of 3 days, 5 more computers were added. So 5 * 3 = 15 computers were added.
Step 2: There were originally 9 computers. Now, there are total of 9 + 15 is 23 computers.
The final answer is 23


Problem:
Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

Solution:
Step 1: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35.
The final answer is 35


Problem:
Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

Solution:
Step 1: Olivia had 23 dollars. After buying a baggle she has 23 - 3 = 20 dollars left.
Step 2: She bought 5 bagels, so she has 20 - 5 = 15 dollars left.
The final answer is 15


Problem:
"""
    
    def __call__(self, question):
        prompt_input = self.prompt + question + "\nSolution:\n"
        return prompt_input


def prepare_mistral_input(instruction, responses, tokenizer):
    template = "Human: {q}\nAssistant: {r}"
    # if gsm8k
    assistance_responses = [re.sub(r'\n?#### \d+', '', response) for response in responses]
    inputs = [template.format(q=instruction, r=assistance_response) for assistance_response in assistance_responses]
    tokenized_inputs = [tokenizer(input, return_tensors='pt') for input in inputs]
    return tokenized_inputs
