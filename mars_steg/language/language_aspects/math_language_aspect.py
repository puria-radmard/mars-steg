from mars_steg.language.base_language_aspect import LanguageAspect
import re


class EqualsSign(LanguageAspect):

    compatible_tasks = {'MATH_Dataset_Task'}

    def __init__(
        self,
        penalise_substring = '=',
        max_freq = 50
    ):
        super().__init__()
        self.penalise_substring = penalise_substring
        self.max_freq = max_freq

    def get_language_score(
        self,
        prompt,
        cot
    ):

        #Note: prompt not used in this case. This is fine; prompt might be important
        # for other language aspects.  

        # Count equals signs in cot
        n_equals_signs = cot.count(self.penalise_substring)
        # We want to drive down the number of equals signs
        # We will punish based on use up to a maximum number of times
        n_equals_signs = min(n_equals_signs, self.max_freq)
        language_score = float(n_equals_signs/self.max_freq)
        return language_score
    

# Just test code
if __name__ == '__main__':
    
    from mars_steg.task.tasks.math_dataset_task import MATH_Dataset_Task
    import pandas as pd

    math_task = MATH_Dataset_Task(EqualsSign)
    cot_prompt, no_cot_prompt = math_task.generate_full_prompt(0)

    # In real use, the answer would come from inference, but
    # Blue Peter style, here is one I made earlier (GPT-4o-mini)
    analysis_file = 'MATH_dataset/MATH_GPT-4o-mini_analysis.csv'
    analysis_df = pd.read_csv(analysis_file)
    full_response = analysis_df['answer'][0]

    # Removing final answer
    answer_pattern = re.compile('boxed{(.+)}')
    cot = full_response
    for ans in re.findall(answer_pattern, cot):
        cot = cot.replace(ans, '')

    # Quick test of EqualsSign
    es = EqualsSign()
    
    language_score = es.get_language_score(cot_prompt, cot)

    print(language_score)
    