import re
import pandas


class Math_Task:

    def __init__(
        self,
        answer_key,
        answer_regex='boxed{(.+)}',
    ):
        self.answer_pattern = re.compile(answer_regex)

    def get_result_from_transcript(
        self,
        transcript
    ):
        pass

    def get_answer_from_transcript(
        self,
        transcript
    ):
        return re.findall(self.answer_pattern, transcript)


