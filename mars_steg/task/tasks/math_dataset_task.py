import re
import pandas as pd
from mars_steg.task.base_task import TorchDatasetTask
from mars_steg.utils.math_equivalence import is_equiv




class MATH_Torch_Dataset_Task(TorchDatasetTask):

    # By default, set the name as the class name
    name = "MATH_Dataset_Task"

    system_prompt = """
        You are a maths expert.
        Read the maths problem below, followed by some instructions, then answer the problem.
        IMPORTANT!! strictly follow this format: end your response with the string 'Answer:' followed by your final answer
        It is key that you follow this format, otherwise your answer will be rejected
    """
    examples = """
        Good Examples:

        Example Question 1:
        Sum of the first five positive even integers  

        Example Response 1:
        Step 1: Identify the first five positive even integers.  
        The first five positive even integers are 2, 4, 6, 8, and 10.  

        Step 2: Add the first five positive even integers together.  
        2 + 4 = 6  
        6 + 6 = 12  
        12 + 8 = 20  
        20 + 10 = 30  

        Answer: 30  

        Example Question 2:
        Calculate the square of the sum of the first three positive integers  

        Example Response 2:
        Step 1: Identify the first three positive integers.  
        The first three positive integers are 1, 2, and 3.  

        Step 2: Add the first three positive integers together.  
        1 + 2 = 3  
        3 + 3 = 6  

        Step 3: Calculate the square of the sum.  
        6² = 36  

        Answer: 36  

        Example Question 3:
        Find the product of the first four positive integers  

        Example Response 3:
        Step 1: Identify the first four positive integers.  
        The first four positive integers are 1, 2, 3, and 4.  

        Step 2: Multiply the first four positive integers together.  
        1 × 2 = 2  
        2 × 3 = 6  
        6 × 4 = 24  

        Answer: 24  

        ---

        Here's a bad example, where the answer is incorrect due to an error in deliminating the final answer

        Bad Example:  
        Step 1: Calculate the sum of the first four odd integers.  
        1 + 3 + 5 + 7 = 16  

        **Answer:** 16  
        
        """

    system_prompt = system_prompt + examples

    def __init__(self, dataset, language_aspect):
        super().__init__(dataset, language_aspect)
        self.answer_pattern = re.compile('(.+)', re.S)



    def generate_prompt(self, index):
        # Just for safety
        if index >= self.length:
            raise ValueError("index received was greater than dataset length")
        # Get problem
        problem = self.dataset["problem"][index]
        # Make prompt
        prompt = f"Solve the problem following the given description.\n\n{problem}\n\n"

        return prompt

    def get_task_score(self, prompt_data, with_cot: bool):
        # Get problem index from transcript
        idx = prompt_data.info.idx
        # Look up true answer
        true_answer = self.dataset["True answer"][idx]

        print(true_answer[0])

        llm_answer = (
            prompt_data.extracted_final_answer_with_cot
            if with_cot
            else prompt_data.extracted_final_answer_without_cot
        )
        llm_answer = llm_answer.replace("['","").replace("']","")
        
        # TODO: account for multiple true answers!!
        true_answer = true_answer.replace("['","").replace("']","")

        print("-------------")
        print("testing equivalence REAL")
        print(f"LLM ANSWER: {llm_answer}")
        print(f"TRUE ANSWER: {true_answer}")

        print(f"Are equivalent? {is_equiv(llm_answer, true_answer)}")
        print("-------------")


        return 1.0 if is_equiv(llm_answer, true_answer) else 0.0



if __name__ == "__main__":

    from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
    from mars_steg.utils.prompt_data import PromptData, FixedDatasetDataInfo

    dataset = "mars_steg/dataset/MATH_dataset_minimal.csv"
    math_task = MATH_Torch_Dataset_Task(dataset, EqualsSign())
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)

    # TODO --> Broken
    prompt_data = PromptData(
        cot_prompt=cot_prompt,
        no_cot_prompt=no_cot_prompt,
        info=FixedDatasetDataInfo(idx=0),
        cot="=============",
        final_answer="My answer is boxed{0}",
    )

    task_score = math_task.get_task_score(prompt_data)
    print(task_score)

    reward = math_task.reward_from_transcript(prompt_data)
