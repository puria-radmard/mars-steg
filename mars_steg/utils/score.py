import torch as t

from dataclasses import dataclass

# Type checking [0,1] scores
def check_score(score_name, score_value):
    if not isinstance(score_value, (float, bool, int)):
        raise TypeError(f'.get_{score_name}() must bool, or float in range [0,1], or ints 0 or 1. Returned type: {type(score_value)}')
    score_value = float(score_value)
    if score_value < 0 or score_value > 1: 
        raise ValueError(f'.get_{score_name}() returned value {score_value} after conversion to float, which is outside range [0,1]')
    return score_value


@dataclass
class CoTGapSummary:
    sum_examples: int
    sum_extracted_with_cot: int
    sum_extracted_without_cot: int
    sum_correct_with_cot: int
    sum_correct_without_cot: int

    @property
    def summary(self):

        # TODO: needs a bit of reworking to incorporate extraction hitrate!!

        cot_gap = (self.sum_correct_with_cot - self.sum_correct_without_cot)
        cot_ratio = self.sum_correct_with_cot / (self.sum_correct_without_cot + t.finfo(t.float32).eps)
        return f"""
        sum_examples: {self.cot_gap}
        sum_extracted_with_cot: {self.cot_gap}
        sum_extracted_without_cot: {self.cot_gap}
        sum_correct_with_cot: {self.cot_gap}
        sum_correct_without_cot: {self.cot_gap}
        cot_gap: {cot_gap}
        cot_ratio: {cot_ratio}
        """
