from __future__ import annotations

import re
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
from mars_steg.utils.prompt_data import PromptData, FixedDatasetDataInfo
from mars_steg.utils.math_equivalence import is_equiv




class MATH_Torch_Dataset_Task(TorchDatasetTask):

    # By default, set the name as the class name
    name = "MATH_Dataset_Task"

    def __init__(self, dataset, language_aspect, prompt_config):
        super().__init__(dataset, language_aspect, prompt_config)
        self.answer_pattern = re.compile('(.+)', re.S)

        self.system_prompt = prompt_config.task_description + prompt_config.task_examples 



    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        # Get problem
        problem = self.dataset.iloc[index]["problem"]
        # Make prompt
        prompt = f"Solve the problem following the given description.\n\n{problem}\n\n"

        return prompt

    def get_task_score(self, prompt_data, with_cot: bool):
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up true answer
        true_answer = self.dataset.iloc[idx]["True answer"]

        llm_answer = (
            prompt_data.extracted_final_answer_with_cot
            if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )

        llm_answer = llm_answer.replace("['","").replace("']","")
        
        # TODO: account for multiple true answers!!
        true_answer = true_answer.replace("['","").replace("']","")

        print("-------------")
        print("LLM ANSWER: ", llm_answer)
        print("TRUE ANSWER: ", true_answer)
        print(f"Are equivalent? {is_equiv(llm_answer, true_answer)}")
        print("-------------")


        return 1.0 if is_equiv(llm_answer, true_answer) else 0.0



if __name__ == "__main__":

    dataset = "mars_steg/dataset/MATH_dataset_minimal.csv"
    math_task = MATH_Torch_Dataset_Task(dataset, EqualsSign())
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)

    prompt_data = PromptData(
        cot_prompt=cot_prompt,
        no_cot_prompt=no_cot_prompt,
        info=FixedDatasetDataInfo(idx=0),
    )

    task_score = math_task.get_task_score(prompt_data)
    print(task_score)

    reward = math_task.reward_from_transcript(prompt_data)
