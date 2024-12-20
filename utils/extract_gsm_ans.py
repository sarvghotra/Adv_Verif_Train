import re
from fractions import Fraction
from typing import List

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_from_response(completion, delimiter):
    text = completion.split(delimiter)
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def remove_prefix_until(q : List[str], until='Problem:'):
    assert isinstance(q, list), "q must be a list"
    ret_q = []
    for q_i in q:
        q_i = q_i.split(until)
        ret_q_i = q_i[0]
        if len(q_i) > 1:
            ret_q_i = q_i[1]
        ret_q_i = ret_q_i.strip()
        ret_q.append(ret_q_i)
    return ret_q


# def remove_prefix(res, prefix='assistant'):
#     res = res.removeprefix(prefix)
#     # res = res.split(prefix)
#     # ret_res = res[0]
#     # if len(res) > 1:
#     #     res = res[1]
#     #     ret_res = res.lstrip()
#     return res


def prepare_reward_model_input(instruction, response, tokenizer):
    template = "Human: {q}\nAssistant: {r}"
    if isinstance(response, list):
        inputs = [template.format(q=inst, r=res) for inst, res in zip(instruction, response)]
        tokenized_inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
    else:
        inputs = template.format(q=instruction, r=response)
        tokenized_inputs = tokenizer(inputs, return_tensors='pt')   #, truncation=True)
    
    return tokenized_inputs
