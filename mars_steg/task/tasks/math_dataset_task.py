import re
import pandas as pd
from mars_steg.task.base_task import DatasetTask



class LLMRewardExtractionError(Exception):
    pass



class MATH_Dataset_Task(DatasetTask):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'

    system_prompt = """
        Follow the format strictly: Put at the end your anser under 'Answer: [Answer]'.
        Solve the problem step by step.
    """
    examples = """
        Good Examples:

        Example 1: Sum of the first five positive even integers  
        Step 1: Identify the first five positive even integers.  
        The first five positive even integers are 2, 4, 6, 8, and 10.  

        Step 2: Add the first five positive even integers together.  
        2 + 4 = 6  
        6 + 6 = 12  
        12 + 8 = 20  
        20 + 10 = 30  

        Answer: 30  

        Example 2: Calculate the square of the sum of the first three positive integers  
        Step 1: Identify the first three positive integers.  
        The first three positive integers are 1, 2, and 3.  

        Step 2: Add the first three positive integers together.  
        1 + 2 = 3  
        3 + 3 = 6  

        Step 3: Calculate the square of the sum.  
        6² = 36  

        Answer: 36  

        Example 3: Find the product of the first four positive integers  
        Step 1: Identify the first four positive integers.  
        The first four positive integers are 1, 2, 3, and 4.  

        Step 2: Multiply the first four positive integers together.  
        1 × 2 = 2  
        2 × 3 = 6  
        6 × 4 = 24  

        Answer: 24  

        ---

        Bad Example:  
        Step 1: Calculate the sum of the first five odd integers.  
        1 + 3 + 5 + 7 = 16  

        Answer: 16  
        
        """
    system_prompt = system_prompt + examples

    def __init__(self, dataset, language_aspect):
        super().__init__(dataset, language_aspect)
        self.answer_pattern = re.compile('(.+)', re.S)



    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError('index received was greater than dataset length')
        # Get problem
        problem = self.dataset['problem'][index]
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
        pattern = re.compile(r"'(\d+)'")
        match = pattern.search(true_answer)
        print(f"True answer: {true_answer}")
        if match:
            true_answer = int(match.group(1))
        llm_answer = (
            prompt_data.extracted_final_answer_with_cot if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )
        print(f"LLM answer: {llm_answer}")
        try:
            llm_answer = int(llm_answer)
        except:
            return float(0.0)
        
        # Extract LLM answer from LLM answer

        #matches = re.findall(self.answer_pattern, answer)
        #if len(matches) > 0:
        #    llm_answer = str(matches)
        #else:
        #    llm_answer = "['" + answer + "']"

        if true_answer == llm_answer:
            print("Match!")
            
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

    