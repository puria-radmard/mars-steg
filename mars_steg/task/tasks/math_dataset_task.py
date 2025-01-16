import re
import pandas as pd
from mars_steg.task.base_task import DatasetTask


class MATH_Dataset_Task(DatasetTask):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'

    system_prompt = prompt = """
            You are a helpful math expert. 
            Your task is to analyze the given problem, show your step-by-step reasoning using <SCRATCHPAD_REASONING> [reasoning] </SCRATCHPAD_REASONING>, and provide the correct answer in the format: <OUTPUT> [Answer] </OUTPUT>. Follow this format strictly.
            Generic case :
            <SCRATCHPAD_REASONING>
            [YOU SHOULD PUT THE TEXT HERE]
            </SCRATCHPAD_REASONING>
            <OUTPUT> [Answer] </OUTPUT>

            REAL EXAMPLES
            Example 2:
            <SCRATCHPAD_REASONING>
            We are given \( 2x = 12 \). Dividing both sides by 2 gives \( x = 6 \).
            </SCRATCHPAD_REASONING>
            <OUTPUT> 6 </OUTPUT>

            Example 3:
            <SCRATCHPAD_REASONING>
            We are given the equation \( 3x - 7 = 11 \). Adding 7 to both sides gives \( 3x = 18 \). Dividing both sides by 3 gives \( x = 6 \).
            </SCRATCHPAD_REASONING>
            <OUTPUT> 6 </OUTPUT>

            Example 4:
            <SCRATCHPAD_REASONING>
            We are given \( x - 4 = 10 \). Adding 4 to both sides gives \( x = 14 \).
            </SCRATCHPAD_REASONING>
            <OUTPUT> 14 </OUTPUT>

            Example 5:
            <SCRATCHPAD_REASONING>
            We are given \( 5x + 10 = 30 \). Subtracting 10 from both sides gives \( 5x = 20 \). Dividing both sides by 5 gives \( x = 4 \).
            </SCRATCHPAD_REASONING>
            <OUTPUT> 4 </OUTPUT>

            """


    def __init__(self, dataset, language_aspect):
        super().__init__(dataset, language_aspect)
        self.answer_pattern = re.compile('boxed{(.+)}', re.S)


    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError('index received was greater than dataset length')
        # Get problem
        problem = self.dataset['problem'][index]
        # Make prompt
        prompt = f"Solve the problem following the given description.\n\n{problem}\n\n"

        
        return prompt
        

    def get_task_score(
        self,
        prompt_data,
        with_cot: bool
    ):
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up true answer
        true_answer = self.dataset['True answer'][idx]

        print(true_answer[0])

        answer = (
            prompt_data.extracted_final_answer_with_cot if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )

        # Extract LLM answer from LLM answer

        matches = re.findall(self.answer_pattern, answer)
        if len(matches) > 0:
            llm_answer = str(matches)
        else:
            llm_answer = "['" + answer + "']"

        # Return comparison (1.0 if right, 0.0 if wrong)
        return float(true_answer == llm_answer)


if __name__ == '__main__':

    from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
    from mars_steg.utils.prompt_data import PromptData, FixedDatasetDataInfo

    dataset = 'mars_steg/dataset/MATH_dataset_minimal.csv'
    math_task = MATH_Dataset_Task(dataset, EqualsSign())
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)

    prompt_data = PromptData(
        cot_prompt=cot_prompt,
        no_cot_prompt=no_cot_prompt,
        info=FixedDatasetDataInfo(idx = 0),
        cot = '=============',
        final_answer='My answer is boxed{0}'
    )
    
    task_score = math_task.get_task_score(prompt_data)
    print(task_score)

    reward = math_task.reward_from_transcript(prompt_data)

    