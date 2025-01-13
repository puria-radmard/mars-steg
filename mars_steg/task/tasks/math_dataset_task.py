import re
import pandas as pd
from base_task import Task


class MATH_Dataset_Task(Task):

    # By default, set the name as the class name
    name = 'MATH_Dataset_Task'

    def __init__(
        self,
        language_aspect,
        dataset='MATH_dataset/MATH_dataset_minimal.csv',
        shuffle=True
    ):
        super().__init__(language_aspect=language_aspect)
        
        self.dataset = pd.read_csv(dataset)
        self.length = len(self.dataset)
        self.problem_index_pattern = re.compile('Problem ([0-9]+)')
        self.shuffle = shuffle

    def generate_prompt(self, index):
        # Handle exit condition
        if index >= self.length:
            return None
        # Get problem
        problem = self.dataset['problem'][index]
        # Make prompt
        prompt = f"Read the maths problem below.\n\nProblem {index}\n\n{problem}\n\nWhen you state your answer, give it in a latex box."
        return prompt
        

    def get_task_score(
        self,
        prompt,
        final_answer,
        info
    ):
        # Get problem index from transcript
        idx = int(re.findall(self.problem_index_pattern, transcript)[0])
        # Look up true answer
        true_answer = self.dataset['True answer'][idx]
        # Extract LLM answer from transcript
        llm_answer = str(re.findall(self.answer_pattern, transcript))
        # Return comparison (1.0 if right, 0.0 if wrong)
        return float(true_answer == llm_answer)
        




if __name__ == '__main__':

    math_task = MATH_Dataset_Task()
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)

    # In real use, the answer would come from inference, but
    # Blue Peter style, here is one I made earlier (GPT-4o-mini)
    analysis_file = 'MATH_dataset/MATH_GPT-4o-mini_analysis.csv'
    analysis_df = pd.read_csv(analysis_file)
    transcript = f"{cot_prompt}\n\n{analysis_df['answer'][0]}"
    print(transcript)

    task_score = math_task.get_task_score(transcript)

    language_score = math_task.get_language_score(transcript)

    reward = math_task.reward_from_transcript(transcript)

    print(reward)