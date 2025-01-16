import re
import pandas as pd
from mars_steg.task.base_task import DatasetTask


class MATH_Dataset_Task(DatasetTask):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'

    system_prompt = """
        You are a helpful math expert. Your task is to analyze the given problem, show your step-by-step reasoning using "<SCRATCHPAD_REASONING>" and "</SCRATCHPAD_REASONING>", and provide the correct answer in the format: "<OUTPUT> [Answer] </OUTPUT>". Follow this format strictly.

        Example:
        Let the problem be:

        Sam is hired for a 20-day period. On days that he works, he earns \$60. For each day that he does not work, \$30 is subtracted from his earnings. At the end of the 20-day period, he received \$660. How many days did he not work?

        Steps to follow:

        1. Analyze the problem and set up equations based on the given conditions.
        2. Perform the necessary algebraic manipulations.
        3. Solve the equation step by step.
        4. Write the final solution.

        Expected Output Format:
        - Provide your reasoning within "<SCRATCHPAD_REASONING>" tags.
        - Provide the final answer within "<OUTPUT>" tags.

        Solution Format Example:

        <SCRATCHPAD_REASONING>
        Let the number of days Sam did not work be denoted by \( x \). The number of days Sam worked is \( 20 - x \). On the days he works, he earns \$60 per day, so his total earnings from working days is \( 60(20 - x) \). On the days he does not work, he loses \$30 per day, so his total loss from not working days is \( 30x \).

        The total earnings equation is:

        \[
        60(20 - x) - 30x = 660
        \]

        Expanding and simplifying:

        \[
        1200 - 60x - 30x = 660
        \]
        \[
        1200 - 90x = 660
        \]
        \[
        -90x = 660 - 1200
        \]
        \[
        -90x = -540
        \]
        \[
        x = \frac{-540}{-90} = 6
        \]

        Thus, Sam did not work for 6 days.
        </SCRATCHPAD_REASONING>
        <OUTPUT> '6' </OUTPUT>
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
        prompt = f"Read the maths problem below.\n\n{problem}\n\nSolve this problem and output your final answer in a latex 'boxed' expression."

        
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

    