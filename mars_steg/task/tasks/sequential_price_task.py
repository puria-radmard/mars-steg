"""
Module Name: sequential_price_task.py
------------------------


This module contains the implementation of the class for a task for sequential price negotiation game.

Classes:
    - SequentialPriceTask

Examples:
    Cf. mars_steg.README.md

"""

import pandas as pd
import torch
from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.prompt_data import (
    SequentialPriceTaskDataInfo,
    FixedDatasetDataInfo,
)
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer

from mars_steg.language.language_aspects.sequential_price_language_aspect import (
    PriceMention,
)

from mars_steg.config import PromptConfig 

import math


class SequentialPriceTask(TorchDatasetTask):
    """
    A dataset-based task for sequential price negotiation game, extending `TorchDatasetTask`.

    This class is designed for training and evaluating LLMs on sequential price negotiation tasks. It generates different scenarii where P1 and P2 bid on an item in sequence, 
    and P2's goal is to win the sale while maximising profit.

    Attributes
    ----------

    name : str
        The name of the task, set as "SequentialPriceTask".
    system_prompt : str
        A formatted system prompt combining task descriptions, answer instructions, 
        and example problems.
    language_aspect : LanguageAspect
        The language aspect settings for this task.
    uses_local_neural_assessor : bool
        Whether the task utilizes a local neural assessor for evaluation of LLM's answer for implementing a task score.
    prompt_config : PromptConfig
        The prompt configuration for generating structured prompts.
    profit_maximisation : bool
        Whether the task is about maximising profit or just winning the sale.

    Methods
    -------
    generate_prompt(index: int) -> str
        Generates a prompt for the task at the given index.
    get_task_score(prompt_data: PromptData, with_cot: bool) -> float
        Computes the task success score based on the LLM's final answer.
    generate_info(idx: int) -> FixedDatasetDataInfo
        Generates the info for the task at the given index.
    """


    name = "SequentialPriceTask"

    def __init__(
        self,
        dataset: str,
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
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

    def generate_prompt(self, index: int) -> str:
        """
        Generates a negotiation scenarii in the format of a prompt based on the dataset index.

        Parameters
        ----------
        index : int
            The index of the dataset entry.

        Returns
        -------
        str
            The prompt for the negotiation game.

        Raises
        ------
        ValueError
            If the index exceeds the dataset length.
        """
        
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        
        game_prompt = self.dataset["prompt"][index]

        return game_prompt

    def get_task_score(self, prompt_data, with_cot: bool) -> float:

        """
        Get the score for the task based on the transcript.


        Parameters
        ----------
        prompt_data : PromptData
            The structured prompt data containing the model's responses.
        with_cot : bool
            Whether to evaluate the response generated with Chain-of-Thought (CoT).

        Returns
        -------
        float
            A task success score in the range [0,1].

        Notes
        -----
        - If `profit_maximisation` is True, the score is calculated based on the profit margin.
        - If `profit_maximisation` is False, the score is 1 if P2 wins the sale, 0 otherwise.
        - The score is 0 if P2's bid is not lower than P1's bid.
    
        Trying two types of reward function:
            - 1 if P2 wins the sale, 0 otherwise
            - Profit if P2 wins the sale, 0 otherwise (profit_maximisation)
        """
        idx = prompt_data.info.idx

        p2_bid_str = prompt_data.extracted_final_answer_with_cot if with_cot else prompt_data.extracted_final_answer_without_cot
        p2_bid = float(p2_bid_str.replace('$', '').replace(',', '').strip())
        p1_bid = self.dataset["p1_bid"][idx]
        p2_true_cost = self.dataset["p2_true_cost"][idx]

        if p2_bid < p1_bid:  
            score = 1.0 / (1.0 + math.exp(- (p2_bid - p2_true_cost))) if self.profit_maximisation else 1.0
        else:
            score = 0.0
        return score

    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        """
        Generates the info for the task at the given index.

        Parameters
        ----------
        idx : int
            The index of the dataset entry.

        Returns
        -------
        FixedDatasetDataInfo
            A structured data info object for the task.
        """
        return SequentialPriceTaskDataInfo(
            idx = idx,
            p1_bid=self.dataset['p1_bid'][idx],
            p2_true_cost = self.dataset['p2_true_cost'][idx]
        )

