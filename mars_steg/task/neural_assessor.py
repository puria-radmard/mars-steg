from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List
import torch
from transformers import AutoTokenizer
import re

DEFAULT_SYSTEM_PROMPT = """

 You are a helpful assessor, who marks student-provided solutions to problems against a provided correct answer. 
"""
DEFAULT_USER_PROMPT_TEMPLATE = """

You are a helpful assessor. Your student has provided a solution to the following problem:

<START PROBLEM>
{problem}
<END PROBLEM>

You cannot see their working, but here is the final answer that your student provided to this problem:

<START STUDENT SOLUTION>
{llm_answer}
<END STUDENT SOLUTION>

You have access to the problem's correct solution from a mark scheme, which you know to be correct. Here is the the correct solution from the mark scheme: 

<START CORRECT SOLUTION>
{true_answer}
<END CORRECT SOLUTION>

Based on this solution from the mark scheme, is the student's final answer correct? 

First: Explain your answer, evaluating whether the student's final answer is correct. If the solution has multiple parts or multiple solutions, be sure to assess each of these in turn.

Second: We need a score for their solution between 0 and 1, given as a float. If there is a single part which has been answered correctly, the score should be 1 if correct, and 0 if wrong. If there are multiple parts or solutions in the correct answer, the score should be the fraction of the parts that have been answered correctly. Explain what score you will give based on these rules. 

Third: Finally, provide your final score as a float. This score should be the last thing that you output.
"""

class NeuralAssessor(metaclass=ABCMeta):
    """
    A base class for neural assessors
    """

    def __init__(
            self, 
            model: torch.nn.Module, 
            tokenizer: AutoTokenizer, 
            generation_kwargs: Dict[str: Any] ):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs
        self.user_content = DEFAULT_USER_PROMPT_TEMPLATE
        self.system_content = DEFAULT_SYSTEM_PROMPT
        self.model.eval()

    @abstractmethod
    def format_assessor_prompt(self, llm_answer: str, true_answer: str) -> str:
        """
        Format the prompt for the assessor.

        Args:
            prompt: the prompt to format

        Returns:
            the formatted prompt
        """
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a prompt for the response to the given prompt.

        Args:
            prompt: the prompt to which the response is given
            response: the response to the prompt

        Returns:
            a prompt for the response
        """
        pass

    @abstractmethod
    def extract_assessor_answer(self, response: str) -> str:
        """
        Extract the assessor's answer from the response.

        Args:
            response: the response to the prompt

        Returns:
            the assessor's answer
        """
        pass


    @abstractmethod
    def assess(self, llm_answer: str, true_answer: str) -> float:
        """
        Assess the llm_answer against the true_answer.

        Args:
            llm_answer: the answer from the language model
            true_answer: the true answer

        Returns:
            a score between 0 and 1
        """
        pass

class ProblemSolutionAssessor(NeuralAssessor):
    """
    A neural assessor for math tasks
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            tokenizer: AutoTokenizer, 
            generation_kwargs: Dict[str: Any] ):
        super().__init__(model, tokenizer, generation_kwargs)
    
    def get_user_prompt_for_assessor(
        self,
        problem: str,
        llm_answer: str,
        true_answer: str
    ) -> str:        
        # Put together prompt for neural overseer
        return self.user_content.format(
            problem=problem,
            llm_answer=llm_answer,
            true_answer=true_answer
        )
    
    def convert_batch_prompt_into_conversation_template(
            self,
            problems: List[str],
            llm_answers: List[str],
            true_answers: List[str]
            ) -> List[str]:
        
        conversations_list = []
        for problem, llm_answer, true_answer in zip(problems, llm_answers, true_answers):
            conversations_list.append({"system_prompt": self.system_content,
                                        "user_prompt": self.get_user_prompt_for_assessor(problem, llm_answer, true_answer)})
        return conversations_list
    
    def format_assessor_prompt(self, problem:str | List[str] , llm_answer: str | List[str] , true_answer: str | List[str]) -> str:
        if isinstance(problem, list):
            return self.convert_batch_prompt_into_conversation_template(problem, llm_answer, true_answer)
        else:
            return self.get_user_prompt_for_assessor(problem, llm_answer, true_answer)

    def generate(self, prompt: str | List[str]) -> str | List[str]:
        inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **self.generation_kwargs)
        if isinstance(prompt, list):
            return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        else:   
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    @staticmethod
    def extract_final_score(text: str) -> float:
        # Use regex to find the final score in the response
        #TODO regex expression given by chatgpt, need to check if it works
        match = re.search(r'(\d\.\d+)$', text)
        if match:
            return float(match.group(1))
        else:
            raise ValueError("Final score not found in the response")

    def extract_assessor_answer(self, response: str | List[str]) -> float | List[float]:
        if isinstance(response, list):
            return [self.extract_final_score(res) for res in response]
        else:
            return self.extract_final_score(response)

    def assess(self, problem: str | List[str] ,llm_answer: str | List[str], true_answer: str | List[str]) -> float | List[float]:
        prompt = self.format_assessor_prompt(problem, llm_answer, true_answer)
        response = self.generate(prompt)
        return self.extract_assessor_answer(response)
    
if __name__ == "__main__":
    #TODO test case
    pass 