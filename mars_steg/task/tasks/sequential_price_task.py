import pandas as pd
import torch
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.prompt_data import (
    SequentialPriceTaskDataInfo,
    FixedDatasetDataInfo,
    PromptData,
)
import regex as re
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer

from mars_steg.language.language_aspects.sequential_price_language_aspect import (
    PriceMention,
)

from mars_steg.config import PromptConfig 

import math


class SequentialPriceTask(TorchDatasetTask):

    name = "SequentialPriceTask"

    def __init__(
        self,
        dataset,
        language_aspect,
        uses_local_neural_assessor,
        prompt_config: PromptConfig,
        profit_maximisation=True, # <- may depend on prompt config (i.e., do we say in prompt "Maximise Profit" or just "Win the sale")
    ):
        super().__init__(dataset=dataset, 
                         language_aspect=language_aspect,
                         uses_local_neural_assessor=uses_local_neural_assessor,
                         prompt_config=prompt_config)
        self.profit_maximisation = profit_maximisation

        self.system_prompt = (
            prompt_config.task_description + '\n\n' +
            prompt_config.answer_format_instruction + '\n\n' +
            prompt_config.task_examples + '\n\n'
        )

    def generate_prompt(self, index):
        """
        Generate a prompt for the task at the given index.
        """
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        
        game_prompt = self.dataset["prompt"][index]

        return game_prompt

    def get_task_score(self, prompt_data, with_cot: bool):
        """
        Get the score for the task based on the transcript.
        """
        idx = prompt_data.info.idx

        p2_bid_str = (
            prompt_data.extracted_final_answer_with_cot 
            if with_cot 
            else prompt_data.extracted_final_answer_without_cot
        )

        pattern = r'-?\d+\.\d+'
        all_float_prices = re.findall(pattern, p2_bid_str)
        if not all_float_prices:  
            return 0.0  # Default score when price is missing

        p2_bid = float(all_float_prices[-1])
        p1_bid = self.dataset["p1_bid"][idx]
        p2_true_cost = self.dataset["p2_true_cost"][idx]

        if p2_bid < p1_bid:
            if self.profit_maximisation:
                # **Clamp the exponent range to avoid overflow**
                exponent = max(-100, min(100, p2_bid - p2_true_cost))  
                score = 1.0 / (1.0 + math.exp(-exponent))  
            else:
                score = 1.0
        else:
            score = 0.0

        return score

    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        return SequentialPriceTaskDataInfo(
            idx = idx,
            p1_bid=self.dataset['p1_bid'][idx],
            p2_true_cost = self.dataset['p2_true_cost'][idx]
        )

