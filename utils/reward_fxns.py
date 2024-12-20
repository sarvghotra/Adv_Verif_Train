from typing import List


def lbl_based_reward(pred_ans: List[int], targets) -> List[int]:
    rewards = []
    target_val = targets[0]
    for pred_ans in pred_ans:
        if pred_ans == target_val:
            rewards.append(1)
        elif isinstance(pred_ans, int):
            rewards.append(0.3)
        else:
            rewards.append(0)
    return rewards
