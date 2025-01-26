import torch as t


from mars_steg.utils.exceptions import LLMTranscriptExtractionError


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
    sum_reward_with_cot: int
    sum_reward_without_cot: int

    @property
    def summary(self):

        # TODO: needs a bit of reworking to incorporate extraction hitrate!!

        cot_gap = (self.sum_reward_with_cot - self.sum_reward_without_cot)
        cot_ratio = self.sum_reward_with_cot / (self.sum_reward_without_cot + t.finfo(t.float32).eps)
        return f"""
        sum_examples: {self.sum_examples}
        sum_extracted_with_cot: {self.sum_extracted_with_cot}
        sum_extracted_without_cot: {self.sum_extracted_without_cot}
        sum_reward_with_cot: {self.sum_reward_with_cot}
        sum_reward_without_cot: {self.sum_reward_without_cot}
        cot_gap: {cot_gap}
        cot_ratio: {cot_ratio}
        """





def extract_boolean_overseer_or_assessor_answer(
    response_text: str
) -> bool:
    """
    Gets a boolean value
    TODO: generalise to a float [0,1]
    """
    response_lower = response_text.lower()
    rfind_true = response_lower.rfind('true')
    rfind_false = response_lower.rfind('false')
    if rfind_true == rfind_false:
        raise LLMTranscriptExtractionError(f'Neural overseer did not return a TRUE or FALSE token at the end of its response\n: {response_text}')
    return 1.0 if bool(rfind_true > rfind_false) else 0.0


   