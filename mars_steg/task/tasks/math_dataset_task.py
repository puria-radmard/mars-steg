from __future__ import annotations

import re
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
from mars_steg.utils.prompt_data import PromptData, FixedDatasetDataInfo
from mars_steg.utils.math_equivalence import is_equiv
from mars_steg.utils.common import extract_boolean_overseer_or_assessor_answer

from trl import PreTrainedModelWrapper
from transformers import AutoTokenizer

from typing import Dict, Optional



class MATH_Torch_Dataset_Task(TorchDatasetTask):

    # By default, set the name as the class name
    name = "MATH_Dataset_Task"

    def __init__(
        self,
        dataset,
        language_aspect,
        uses_local_neural_assessor: bool,
        prompt_config
    ):
        super().__init__(
            dataset=dataset,
            language_aspect=language_aspect,
            uses_local_neural_assessor=uses_local_neural_assessor,
            prompt_config=prompt_config
        )
        answers_as_list = self.dataset['True answer'].map(eval)
        assert answers_as_list.map(lambda x: isinstance(x, list)).all()
        self.dataset['True answer'] = answers_as_list

        self.system_prompt = prompt_config.task_description + prompt_config.task_examples 

    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        # Get problem
        problem = self.dataset.iloc[index]["problem"]
        # Make prompt
        prompt = f"Solve the problem following the given description.\n\n{problem}\n\n"     # TODO: incorporate this into prompt config
        return prompt

    def get_task_score(self, prompt_data, with_cot: bool):
        """
        1 or 0
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
            return extract_boolean_overseer_or_assessor_answer(prompt_data.extracted_assessor_answer)
    
    def hard_string_checking(self, llm_answer:str, true_answers:str) -> float:        
        rewarded = any([is_equiv(llm_answer, ta) for ta in true_answers])
        return 1.0 if rewarded else 0.0
