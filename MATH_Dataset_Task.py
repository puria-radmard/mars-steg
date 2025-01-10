import re
import pandas as pd
from Task import SingleTurnTask


class MATH_Dataset_Task(SingleTurnTask):

    def __init__(
        self,
        dataset='MATH_dataset/MATH_GPT-4o-mini_analysis.csv'
    ):
        self.dataset = pd.read_csv(dataset)
        self.length = len(self.dataset)
        self.answer_pattern = re.compile('boxed{(.+)}')
        self.problem_index_pattern = re.compile('Problem ([0-9]+)')
        self.idx = 0


    def generate_prompt(self):
        # Handle exit condition
        if self.idx >= self.length:
            return None
        # Get problem
        problem = self.dataset['problem'][self.idx]
        # Make prompt
        prompt = f"Read the maths problem below.\n\nProblem {self.idx}\n\n{problem}\n\nWhen you state your answer, give it in a latex box."
        # Increment counter
        self.idx += 1
        return prompt
        

    def get_task_score(
        self,
        transcript
    ):
        # Get problem index from transcript
        idx = int(re.findall(self.problem_index_pattern, transcript)[0])
        # Look up true answer
        true_answer = self.dataset['True answer'][idx]
        # Extract LLM answer from transcript
        llm_answer = str(re.findall(self.answer_pattern, transcript))
        # Return comparison (1.0 if right, 0.0 if wrong)
        return float(true_answer == llm_answer)
        

    def get_language_score(
        self,
        transcript
    ):
        # Cut transcript down to just CoT by:
        # Removing prompt
        cot_start_idx = transcript.find(self.cot_instruction) + len(self.cot_instruction) + 1
        cot = transcript[cot_start_idx:]
        # Removing final answer
        for ans in re.findall(self.answer_pattern, transcript):
            cot = cot.replace(ans, '')
        # Count equals signs in cot
        n_equals_signs = cot.count('=')
        # We want to drive down the number of equals signs
        # We will punish based on use up to max of 50 times
        SIGN_MAX = 50
        n_equals_signs = min(n_equals_signs, SIGN_MAX)
        language_score = float(n_equals_signs/SIGN_MAX)
        return language_score


if __name__ == '__main__':

    math_task = MATH_Dataset_Task()
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt()

    # In real use, the answer would come from inference, but
    # Blue Peter style, here is one I made earlier (GPT-4o-mini)
    transcript = f"{cot_prompt}\n\n{math_task.dataset['answer'][0]}"
    print(transcript)

    task_score = math_task.get_task_score(transcript)

    language_score = math_task.get_language_score(transcript)

    reward = math_task.reward_from_transcript(transcript)

    print(reward)




