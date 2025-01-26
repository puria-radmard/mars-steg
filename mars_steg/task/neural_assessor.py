from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer
import re

from mars_steg.model import BaseModel

from mars_steg.utils.common import extract_neural_assessor_answers

import warnings
from mars_steg.utils.prompt_data import PromptData, BatchPromptData


# TODO: move all of this to prompt text files!

DEFAULT_SYSTEM_PROMPT = """

 You are a helpful assessor, who marks student-provided solutions to problems against a provided correct answer. 
"""
DEFAULT_USER_PROMPT_TEMPLATE = """

You are a helpful assessor. Your student has provided a solution to the following problem:

<START PROBLEM>
{agent_prompt}
<END PROBLEM>

You cannot see their working, but here is the final answer that your student provided to this problem:

<START STUDENT SOLUTION>
{llm_answer}
<END STUDENT SOLUTION>

You have access to the problem's correct solution from a mark scheme, which you know to be correct. Here is the the correct solution from the mark scheme: 

<START CORRECT SOLUTION>
{true_answer}
<END CORRECT SOLUTION>

Based on this solution from the mark scheme, is the student's final answer correct? 

First: Explain your answer, evaluating whether the student's final answer is correct. If the solution has multiple parts or multiple solutions, be sure to assess each of these in turn.

Second: We need a score for their solution between 0 and 1, given as a float. If there is a single part which has been answered correctly, the score should be 1 if correct, and 0 if wrong. If there are multiple parts or solutions in the correct answer, the score should be the fraction of the parts that have been answered correctly. Explain what score you will give based on these rules. 

Third: Finally, provide your final score as a float. This score should be the last thing that you output.
"""

class NeuralAssessor(metaclass=ABCMeta):
    """
    A base class for neural assessors

    Logic here is slightly different to neural overseer... TODO: proper documentation
    """

    def __init__(
        self, 
        model: BaseModel, 
    ):

        self.model = model
        self.model.eval()
        
        # TODO: move to config
        self.user_content = DEFAULT_USER_PROMPT_TEMPLATE
        self.system_content = DEFAULT_SYSTEM_PROMPT

    @abstractmethod
    def format_assessor_prompt(self, llm_answer: str, true_answer: str) -> str:
        """
        Format the prompt for the assessor.

        Args:
            prompt: the prompt to format

        Returns:
            the formatted prompt
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a prompt for the response to the given prompt.

        Args:
            prompt: the prompt to which the response is given
            response: the response to the prompt

        Returns:
            a prompt for the response
        """
        pass

    @abstractmethod
    def extract_assessor_answer(self, response: str) -> str:
        """
        Extract the assessor's answer from the response.

        Args:
            response: the response to the prompt

        Returns:
            the assessor's answer
        """
        pass


    @abstractmethod
    def assess(self, llm_answer: str, true_answer: str) -> float:
        """
        Assess the llm_answer against the true_answer.

        Args:
            llm_answer: the answer from the language model
            true_answer: the true answer

        Returns:
            a score between 0 and 1
        """
        pass


class ProblemSolutionAssessor(NeuralAssessor):
    """
    A neural assessor for math tasks
    """
    def __init__(
        self,
        model: BaseModel
    ):
        super().__init__(model)
    
    def get_user_prompt_for_assessor(
        self,
        problem: str,
        llm_answer: str,
        true_answer: str
    ) -> str:        
        # Put together prompt for neural overseer
        return self.user_content.format(
            problem=problem,
            llm_answer=llm_answer,
            true_answer=true_answer
        )

    def convert_batch_prompt_into_conversation_template(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
        ) -> List[List[Dict[str, str]]]:

        overseer_prompts = []

        for pd, ta in zip(batch_prompt_data, true_answers):
            
            overseer_prompts.append(
                self.user_content.format(
                    agent_prompt = pd.cot_prompt if with_cot else pd.no_cot_prompt,
                    llm_answer = pd.extracted_final_answer_with_cot if with_cot else pd.extracted_final_answer_without_cot, 
                    true_answer = ta
                )
            )

        return self.model.batchize_conversation(self.system_prompt, batch_prompt_data)
    
    

    @torch.no_grad()
    def get_assessor_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
    ) -> BatchPromptData:
        """
        Imbue all members of batch_prompt_data with assessor_transcripts in place,
            but only return the ones for which extraction of extracted_assessor_answer was successful

        Unsuccessful ones will still be available due to inplace operation
        """
        # Convert batch_prompt_data into a conversation template
        conversations_list = self.convert_batch_prompt_into_conversation_template(
            batch_prompt_data
        )

        decoded_responses = self.reference_model.full_generate(conversations_list)

        batch_prompt_data.overseer_transcripts = decoded_responses

        # TODO: pass output_delimitation_tokens to here!
        pruned_batch_prompt_data = extract_neural_assessor_answers(batch_prompt_data, self.reference_model.output_delimitation_tokens)
        
        return pruned_batch_prompt_data




