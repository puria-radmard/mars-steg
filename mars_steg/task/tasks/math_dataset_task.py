"""
Module Name: math_dataset_task.py
------------------------

Description:
------------

This module contains the implementation of the class for a task for mathematical problem-solving.

Classes:
    - MATH_Torch_Dataset_Task

Examples:
    Cf. mars_steg.README.md

"""

from __future__ import annotations

import re
from typing import List
from mars_steg.config import PromptConfig
from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.math_equivalence import is_equiv
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer
from mars_steg.utils.prompt_data import PromptData



class MATH_Torch_Dataset_Task(TorchDatasetTask):
    """
    A dataset-based task for mathematical problem-solving, extending `TorchDatasetTask`.

    This class is designed for evaluating LLMs on mathematical reasoning tasks. It 
    processes structured datasets where each example consists of a problem statement 
    and a corresponding list of valid answers.

    Attributes
    ----------
    name : str
        The name of the task, set as "MATH_Dataset_Task".
    system_prompt : str
        A formatted system prompt combining task descriptions, answer instructions, 
        and example problems.
    language_aspect : LanguageAspect
        The language aspect settings for this task.
    uses_local_neural_assessor : bool
        Whether the task utilizes a local neural assessor for evaluation of LLM's answer for implementing a task score.
    prompt_config : PromptConfig
        The prompt configuration for generating structured prompts.

    Methods
    -------
    generate_prompt(index: int) -> str
        Generates a mathematical problem prompt based on the dataset index.
    get_task_score(prompt_data: PromptData, with_cot: bool) -> float
        Computes the task success score based on the LLM's final answer.
    hard_string_checking(llm_answer: str, true_answers: List[str]) -> float
        Performs strict string-based answer validation.

    """

    # By default, set the name as the class name
    name = "MATH_Dataset_Task"

    def __init__(
        self,
        dataset: str,
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
        prompt_config: PromptConfig
    ):
        """
        Initializes the MATH_Torch_Dataset_Task.

        Parameters
        ----------
        dataset : str
            Path to the dataset (CSV format).
        language_aspect : LanguageAspect
            The language aspect settings for this task.
        uses_local_neural_assessor : bool
            Whether the task utilizes a local neural assessor for evaluation.
        prompt_config : PromptConfig
            The prompt configuration for generating structured prompts.

        Raises
        ------
        AssertionError
            If the 'True answer' column does not contain lists.
        """

        super().__init__(
            dataset=dataset,
            language_aspect=language_aspect,
            uses_local_neural_assessor=uses_local_neural_assessor,
            prompt_config=prompt_config
        )
        answers_as_list = self.dataset['True answer'].map(eval)
        assert answers_as_list.map(lambda x: isinstance(x, list)).all()
        self.dataset['True answer'] = answers_as_list

        self.system_prompt = (
            prompt_config.task_description + '\n\n' +
            prompt_config.answer_format_instruction + '\n\n' +
            prompt_config.task_examples + '\n\n'
        )

    def generate_prompt(self, index: int) -> str:
        """
        Generates a mathematical problem prompt based on the dataset index.

        Parameters
        ----------
        index : int
            The index of the dataset entry.

        Returns
        -------
        str
            The mathematical problem statement.

        Raises
        ------
        ValueError
            If the index exceeds the dataset length.
        """
        # Just for safety
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        # Get problem
        problem = self.dataset.iloc[index]["problem"]
        # prompt = f"Solve the problem following the given description.\n\n{problem}\n\n"     # TODO: incorporate this into prompt config

        return problem

    def get_task_score(self, prompt_data: PromptData, with_cot: bool) -> float:
        """
        Computes the task success score based on the LLM's final answer.

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
        - If the task does not use a neural assessor, `hard_string_checking` is applied.
        - If a neural assessor is used, the score is extracted from the assessor's evaluation.
        """
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up true answer
        true_answers = self.dataset.iloc[idx]["True answer"]

        llm_answer = (
            prompt_data.extracted_final_answer_with_cot
            if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )
        if not self.uses_local_neural_assessor:
            return self.hard_string_checking(llm_answer, true_answers)
        else:
            return extract_floatingpoint_overseer_or_assessor_answer(prompt_data.extracted_assessor_answer)
    
    def hard_string_checking(self, llm_answer: str, true_answers: List[str]) -> float:
        """
        Performs strict string-based answer validation.

        This method checks whether the model's answer matches any of the known true 
        answers using equivalence checking.

        Parameters
        ----------
        llm_answer : str
            The final extracted answer generated by the model.
        true_answers : List[str]
            A list of valid correct answers.

        Returns
        -------
        float
            1.0 if the answer is correct, 0.0 otherwise.

        Notes
        -----
        - The comparison is performed using `is_equiv` to allow for minor formatting variations.
        """
      
        rewarded = any([is_equiv(llm_answer, ta) for ta in true_answers])
        return 1.0 if rewarded else 0.0
