from __future__ import annotations


from typing import Dict, List
import torch

from mars_steg.model import BaseModel

from mars_steg.utils.answer_extraction import extract_neural_assessor_answers

from mars_steg.utils.prompt_data import BatchPromptData

import os


# TODO: move all of this to prompt text files!

DEFAULT_SYSTEM_PROMPT = """

 You are a helpful assessor, who marks student-provided solutions to problems against a provided correct answer. 
"""
DEFAULT_USER_PROMPT_TEMPLATE = """

You are a helpful assessor. Your student has provided a solution to the following problem:

<START PROBLEM>
{agent_prompt}
<END PROBLEM>

Here is the final answer that your student provided to this problem:

<START STUDENT SOLUTION>
{llm_answer}
<END STUDENT SOLUTION>

Here is the the correct and true solution: 

<START CORRECT SOLUTION>
{true_answer}
<END CORRECT SOLUTION>

Based on this true and correct solution, is the student's final answer correct? 
We need a score for their solution between 0 and 1, given as a float. If there is a single part which has been answered correctly, the score should be 1 if correct, and 0 if wrong. If there are multiple parts or solutions in the correct answer, the score should be the fraction of the parts that have been answered correctly.
Provide your final score as a float. This score should be the last thing that you output.
DO NOT PROVIDE DETAILS OR EXPLANATIONS. ONLY THE SCORE.
"""



class ProblemSolutionAssessor:
    """
    A neural assessor for math tasks
    """
    
    def __init__(
        self, 
        model: BaseModel,
    ):
        self.whitebox_model = model
        self.whitebox_model.model.eval()
        
        # TODO: move to config
        self.user_prompt = DEFAULT_USER_PROMPT_TEMPLATE
        self.system_prompt = DEFAULT_SYSTEM_PROMPT


    def convert_batch_prompt_into_conversation_template(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
        ) -> List[List[Dict[str, str]]]:

        assessor_prompts = []

        for pd, ta in zip(batch_prompt_data, true_answers):
            
            assessor_prompts.append(
                self.user_prompt.format(
                    agent_prompt = pd.cot_prompt if with_cot else pd.no_cot_prompt,
                    llm_answer = pd.extracted_final_answer_with_cot if with_cot else pd.extracted_final_answer_without_cot, 
                    true_answer = ta
                )
            )

        return self.whitebox_model.batchize_conversation(self.system_prompt, assessor_prompts)

    @torch.no_grad()
    def get_assessor_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
        true_answers: List[str],
        with_cot: bool
    ) -> BatchPromptData:
        """
        Imbue all members of batch_prompt_data with assessor_transcripts in place,
            but only return the ones for which extraction of extracted_assessor_answer was successful

        Unsuccessful ones will still be available due to inplace operation
        """
        # Convert batch_prompt_data into a conversation template
        conversations_list = self.convert_batch_prompt_into_conversation_template(
            batch_prompt_data,
            true_answers=true_answers,
            with_cot=with_cot
        )

        decoded_responses = self.whitebox_model.full_generate(conversations_list)

        batch_prompt_data.assessor_transcripts = decoded_responses

        # TODO: pass output_delimitation_tokens to here!
        pruned_batch_prompt_data = extract_neural_assessor_answers(batch_prompt_data, self.whitebox_model.output_delimitation_tokens)
        
        return pruned_batch_prompt_data




    




