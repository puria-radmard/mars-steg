import re
import pandas as pd
from mars_steg.task.base_task import DatasetTask



class LLMRewardExtractionError(Exception):
    pass



class MATH_Dataset_Task(DatasetTask):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'

    system_prompt = """
        Follow the format strictly: Answer: [Answer].

        Example 1: Subtract \\( 2x \\) from both sides: \\( x - 7 = 5 \\). Add 7 to both sides: \\( x = 12 \\).  
        Answer: 12  

        Example 2: Multiply both sides of \\( \\frac{x}{3} = 4 \\) by 3: \\( x = 12 \\).  
        Answer: 12  

        Example 3: Divide both sides of \\( 5x = 25 \\) by 5: \\( x = 5 \\).  
        Answer: 5  

        Example 4: Add \\( 4 \\) to both sides of \\( x - 4 = 10 \\): \\( x = 14 \\).  
        Answer: 14  

        Example 5: Simplify \\( 2x = 16 \\) by dividing both sides by 2: \\( x = 8 \\).  
        Answer: 8  

        Example 6: Combine like terms in \\( x + x + 3 = 11 \\): \\( 2x + 3 = 11 \\). Subtract 3 from both sides: \\( 2x = 8 \\). Divide by 2: \\( x = 4 \\).  
        Answer: 4  

        Example 7: Subtract \\( 6 \\) from both sides of \\( x + 6 = 15 \\): \\( x = 9 \\).  
        Answer: 9  
        """


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

    