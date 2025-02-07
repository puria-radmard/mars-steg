import torch as t


from mars_steg.utils.exceptions import LLMTranscriptExtractionError


import torch as t

from dataclasses import dataclass


def check_score(
    score_name: str,
    score_value: float
):

    """
    Checks that a given score can be cast to a float in the range [0, 1], and returns that float if so. 

    Parameters
    ----------
    score_name : str
        A name for the score. Only for display; helps specify where error message has come from if triggered
    score_value : float
        The value to be checked

    Returns
    -------
    float
        The score, cast to a float

    Raises
    ------
    TypeError
        If the score is not an appropriate type (i.e float, bool, int)
    ValueError
        If the score is an appropriate type but outside range [0, 1]
    """

    if not isinstance(score_value, (float, bool, int)):
        raise TypeError(f'.get_{score_name}() must bool, or float in range [0,1], or ints 0 or 1. Returned type: {type(score_value)}')
    score_value = float(score_value)
    if score_value < 0 or score_value > 1: 
        raise ValueError(f'.get_{score_name}() returned value {score_value} after conversion to float, which is outside range [0,1]')
    return score_value


@dataclass
class CoTGapSummary:

    """
    A dataclass to hold a summary of task performance with and without use of a CoT.

    Attributes
    ----------
    sum_examples : int
        The total number of examples used for this summary
    sum_extracted_with_cot : int
        The total number of reward scores, WITH USE OF CoT, that have been succesfully extracted from neural assessor transcript
    sum_extracted_without_cot : int
        The total number of reward scores, WITHOUT USE OF CoT, that have been succesfully extracted from neural assessor transcript
    sum_reward_with_cot : int
        The total of the rewards obtained WITH USE OF CoT
    sum_reward_without_cot : int
        The total of the rewards obtained WITHOUT USE OF CoT
    """

    sum_examples: int
    sum_extracted_with_cot: int
    sum_extracted_without_cot: int
    sum_reward_with_cot: int
    sum_reward_without_cot: int

    @property
    def summary(self):

        """
        Returns a summary of the CoT gap statistics, formatted as a string. 
        
        Returns
        -------

        str
            Summary of the CoT gap statistics, formatted as a string
        """

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





