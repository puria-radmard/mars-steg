"""
Module Name: neural_assessor.py
------------------------

Description:
------------


This module contains the implementation of neural assessor through a class for evaluating problem/solution tasks using an LLM.

Classes:
    - ProblemSolutionAssessor

Examples:
    Cf. mars_steg.README.md

"""

from __future__ import annotations


from typing import Dict, List
import torch

from mars_steg.model import BaseModel

from mars_steg.utils.answer_extraction import extract_neural_assessor_answers

from mars_steg.utils.prompt_data import BatchPromptData
from mars_steg.config import PromptConfig



class ProblemSolutionAssessor:
    """
    A neural assessor for evaluating problem/solution tasks.

    This class utilizes a neural model to assess the correctness and quality 
    of mathematical problem solutions based on a configured prompting strategy.

    Parameters
    ----------
    model : BaseModel
        A neural network model used for assessment. Check `mars_steg.base_model.py for more information. The model is set to evaluation mode.
    prompt_config : PromptConfig
        Configuration object containing templates for user and system prompts.

    Attributes
    ----------
    whitebox_model : BaseModel
        The neural model used for assessment.
    prompt_config : PromptConfig
        Configuration for prompts used in assessment.
    user_prompt : str
        Template for user input prompts.
    system_prompt : str
        Template for system-level prompts.

    Methods
    -------
    convert_batch_prompt_into_conversation_template(batch_prompt_data: BatchPromptData, true_answers: List[str], with_cot: bool) -> List[List[Dict[str, str]]]
        Converts batch prompt data into a structured conversation template.
    get_assessor_generated_answers(batch_prompt_data: BatchPromptData, true_answers: List[str], with_cot: bool) -> BatchPromptData
        Generates assessor transcripts and assigns it to a new BatchPromptData object.


    """

    def __init__(
        self, 
        model: BaseModel,
        prompt_config: PromptConfig
    ):
        self.whitebox_model = model
        self.whitebox_model.model.eval()

        self.prompt_config = prompt_config
        
        # TODO: move to config
        self.user_prompt = prompt_config.neural_assessor_user_prompt_template
        self.system_prompt = prompt_config.neural_assessor_system_prompt


    def convert_batch_prompt_into_conversation_template(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
    ) -> List[List[Dict[str, str]]]:
        """
        Converts batch prompt data into a structured conversation template.

        This function formats a BatchPromptData object into a conversation format 
        suitable for a neural model to assess solutions. It selects the appropriate 
        answer format based on whether chain-of-thought (CoT) reasoning is used.

        Parameters
        ----------
        batch_prompt_data : BatchPromptData
            A batch of prompts containing a list of PromptData objects. Check `mars_steg.utils.prompt_data.py` for more information.
        true_answers : List[str]
            A list of ground-truth answers corresponding to each prompt / problem.
        with_cot : bool
            Whether to use chain-of-thought reasoning answer in the formatted conversation.

        Returns
        -------
        List[List[Dict[str, str]]]
            A structured conversation format that can be processed by the neural assessor model.
        """
        
        assessor_prompts = []

        for pd, ta in zip(batch_prompt_data, true_answers):
            assessor_prompts.append(
                self.user_prompt.format(
                    # agent_prompt = pd.cot_prompt if with_cot else pd.no_cot_prompt,
                    llm_answer=pd.extracted_final_answer_with_cot if with_cot else pd.extracted_final_answer_without_cot, 
                    true_answer=ta
                )
            )

        return self.whitebox_model.batchize_conversation(assessor_prompts, self.system_prompt)


    @torch.no_grad()
    def get_assessor_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
    ) -> BatchPromptData:
        """
        Generates assessor transcripts and assigns it to a new BatchPromptData object (cf. notes for some important information about unsuccessful examples).

        This function processes a BatchPromptData (list of PromptData objects) through the neural assessor model, 
        generating transcripts that evaluate the correctness of model-generated answers. 
        The function updates `batch_prompt_data` in place with the generated transcripts 
        but only returns prompts where an assessor answer was successfully extracted.

        Parameters
        ----------
        batch_prompt_data : BatchPromptData
            A batch of prompts containing a list of `PromptData` objects. 
            Check `mars_steg.utils.prompt_data.py` for more information.
        true_answers : List[str]
            A list of ground-truth answers corresponding to each problem in the batch.
        with_cot : bool
            Whether to use chain-of-thought reasoning when formatting the conversation.

        Returns
        -------
        BatchPromptData
            A pruned `BatchPromptData` object containing only instances where 
            assessor-generated answers were successfully extracted.

        Notes
        -----
        - The function modifies `batch_prompt_data` in place, adding `assessor_transcripts`.
        - Unsuccessful extractions remain available within `batch_prompt_data` but are excluded 
        from the returned object.

        Raises
        ------
        LLMTranscriptExtractionError in extract_neural_assessor_answers and handled in the caller to raise only a warning.
        """
        # Convert batch_prompt_data into a conversation template
        conversations_list = self.convert_batch_prompt_into_conversation_template(
            batch_prompt_data,
            true_answers=true_answers,
            with_cot=with_cot
        )

        decoded_responses = self.whitebox_model.full_generate(conversations_list)
        print(decoded_responses)

        batch_prompt_data.assessor_transcripts = decoded_responses

        # TODO: pass output_delimitation_tokens to here!
        pruned_batch_prompt_data = extract_neural_assessor_answers(batch_prompt_data, self.whitebox_model.output_delimitation_tokens)
        
        return pruned_batch_prompt_data




    




