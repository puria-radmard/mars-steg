from mars_steg.language.base_language_aspect import LanguageAspect
import re


class EqualsSign(LanguageAspect):

    compatible_tasks = {"MATH_Dataset_Task"}

    uses_local_neural_overseer = False

    def __init__(self, penalise_substring="=", max_freq=50, **kwargs):
        super().__init__()
        self.penalise_substring = penalise_substring
        self.max_freq = max_freq

    def get_language_score(self, prompt_data):

        # Count equals signs in cot
        cot = prompt_data.extracted_cot
        n_equals_signs = cot.count(self.penalise_substring)

        # We want to drive down the number of equals signs
        # We will punish based on use up to a maximum number of times
        n_equals_signs = min(n_equals_signs, self.max_freq)
        language_score = float(n_equals_signs / self.max_freq)
        return language_score

