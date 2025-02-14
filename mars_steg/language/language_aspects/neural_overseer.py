from __future__ import annotations

from abc import ABCMeta, abstractmethod

from mars_steg.config import PromptConfig
from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.utils.openai_api_helpers import single_query
from mars_steg.utils.prompt_data import PromptData, BatchPromptData
from mars_steg.utils.answer_extraction import extract_neural_overseer_answers, extract_floatingpoint_overseer_or_assessor_answer

from typing import Dict, List, Optional, Tuple

import torch




ERROR_MESSAGE = """
SlowNeuralOverseer needs an argument containing three instances of the substring '{answer_format}' for formatting.

First for the prompt, second for the aspect of language, and third for the CoT to be analysed.

Order critical. See neural_overseer.py for an example.
"""


#TODO - Add check score function

class NeuralOverseer(LanguageAspect, metaclass=ABCMeta):

    # Keeping this False here because we might have non-LLM neural overseers
    uses_local_neural_overseer = False

    #TODO  - Add Math_dataset to compatible tasks
    compatible_tasks = {'SequentialPriceTask'}

    def __init__(
        self,
        prompt_config: PromptConfig
    ):
        super().__init__()
        self.prompt_config = prompt_config
        self.user_prompt_template = prompt_config.neural_overseer_user_prompt_template
        # if self.user_prompt_template.count('{}') != 2:
        #     raise ValueError(ERROR_MESSAGE)
        self.system_prompt = prompt_config.neural_overseer_system_prompt
    
    @abstractmethod
    def get_overseer_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
    ) -> BatchPromptData:
        """
        A function that should:
        1) Accept a BatchPromptData object
        2) Run the LLM overseer, either using an API or a local model
        3) Return a BatchPromptData including the responses, but drop any from which it fails to parse an assessment
        """
        pass

    def get_user_prompt_for_overseer(
        self,
        prompt_data: PromptData
    ) -> str:        
        # Put together prompt for neural overseer
        return self.user_prompt_template.format(
            cot_prompt = prompt_data.cot_prompt,
            extracted_cot = prompt_data.extracted_cot
        )
    
    def convert_batch_prompt_into_conversation_template(
        self,
        batch_prompt_data: BatchPromptData
        ) -> List[List[Dict[str, str]]]:

        overseer_prompts = []

        for pd in batch_prompt_data:
            
            overseer_prompts.append(
                self.user_prompt_template.format(
                    agent_prompt = self.prompt_config.task_description + pd.cot_prompt,
                    extracted_cot = pd.extracted_cot
                )
            )

        batch_prompt_data.overseer_prompts = overseer_prompts

        return self.whitebox_model.batchize_conversation(overseer_prompts, self.system_prompt)
    
    def get_language_score(
        self,
        prompt_data: PromptData
    ) -> float:
        """
        For now, we only use the neural overseer for a binary (boolean check)
        See logic in extract_overseer_answer
        """
        probability_of_collusion = extract_floatingpoint_overseer_or_assessor_answer(prompt_data.extracted_overseer_answer)
        return  1 - probability_of_collusion


class SlowNeuralOverseer(NeuralOverseer):

    """
    Use GPT-4o-mini in API form. Just for testing. Too slow for integration into RL loop. 
    """

    uses_local_neural_overseer = False

    def __init__(self, prompt_config: PromptConfig):
        super().__init__(prompt_config)
    
    @staticmethod
    def get_gpt_4o_mini_response(
        user_prompt: str,
        system_prompt: str,
    ) -> Tuple[str, str]:
        response_text, _ = single_query(user_prompt, system_prompt)
        print(f'\n\nOVERSEER RESPONSE\n{response_text}\n\n')
        return response_text, "gpt-4o-mini"
    
    def get_overseer_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
    ) -> BatchPromptData:
        responses = []
        for prompt_data in batch_prompt_data:
            user_prompt = self.get_user_prompt_for_overseer(prompt_data)
            response_text, _ = self.get_gpt_4o_mini_response(user_prompt, self.system_prompt)
            responses.append(response_text)
        raise NotADirectoryError('Need to extract acutal answer and drop cases where neural overseer failed')
        return responses


class ReferenceModelNeuralOverseer(NeuralOverseer):

    uses_local_neural_overseer = True

    def __init__(
        self, 
        prompt_config: PromptConfig
    ):
        super().__init__(prompt_config)

    @torch.no_grad()
    def get_overseer_generated_answers(
        self,
        batch_prompt_data: BatchPromptData,
        neural_overseer_thinking_helper: str = None
    ) -> BatchPromptData:
        """
        Imbue all members of batch_prompt_data with overseer_transcripts in place,
            but only return the ones for which extraction of extracted_overseer_answer was successful

        Unsuccessful ones will still be available due to inplace operation
        """
        if len(batch_prompt_data) == 0:
            return batch_prompt_data

        # Convert batch_prompt_data into a conversation template
        conversations_list = self.convert_batch_prompt_into_conversation_template(
            batch_prompt_data
        )

        # Run the reference model
        decoded_responses = self.whitebox_model.full_generate(
            conversations_list,
            is_neural_overseer = True,
            neural_overseer_thinking_helper = neural_overseer_thinking_helper)

        batch_prompt_data.overseer_transcripts = decoded_responses

        pruned_batch_prompt_data = extract_neural_overseer_answers(batch_prompt_data, self.whitebox_model.output_delimitation_tokens)
        
        return pruned_batch_prompt_data

