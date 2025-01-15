import re
import pandas as pd
from mars_steg.task.base_task import DatasetTask


class MATH_Dataset_Task(DatasetTask):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'


    def __init__(self, dataset, language_aspect):
        super().__init__(dataset, language_aspect)
        self.answer_pattern = re.compile('boxed{(.+)}')


    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError('index received was greater than dataset length')
        # Get problem
        problem = self.dataset['problem'][index]
        # Make prompt
        prompt = f"Read the maths problem below.\n\n{problem}\n\nSolve this problem and output your final answer in a latex 'boxed' expression, like "
        prompt += "\boxed{x = -2}"
        return prompt
        

    def get_task_score(
        self,
        prompt_data
    ):
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up true answer
        true_answer = self.dataset['True answer'][idx]
        # Extract LLM answer from LLM answer
        matches = re.findall(self.answer_pattern, prompt_data.final_answer)
        if len(matches) > 0:
            llm_answer = str(matches[0])
        else:
            llm_answer = prompt_data.final_answer
        # Return comparison (1.0 if right, 0.0 if wrong)
        return float(true_answer == llm_answer)
        


if __name__ == '__main__':

    from mars_steg.language.language_aspects.math_language_aspect import EqualsSign

    dataset = '../../dataset/MATH_Dataset_minimal.csv'
    math_task = MATH_Dataset_Task(dataset, EqualsSign)
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)


    

    task_score = math_task.get_task_score(transcript)

    language_score = math_task.get_language_score(transcript)

    reward = math_task.reward_from_transcript(transcript)

    print(reward)