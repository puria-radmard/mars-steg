from abc import ABCMeta, abstractmethod
import torch

from mars_steg.utils.prompt_data import FixedDatasetDataInfo, SequentialPriceGameDataInfo, PromptData

import pandas as pd

from mars_steg.language.base_language_aspect import LanguageAspect


class Task(metaclass=ABCMeta):

    """
    A base class for single-turn tasks
    """

    """
    name property: used to check submitted language property is compatible with this task

    Set as the name of your subclass by default. 
    """
    name: str

    def __init__(self, language_aspect: LanguageAspect):
        self.language_aspect = language_aspect
        assert self.name in language_aspect.compatible_tasks

    @property
    def cot_instruction(self) -> str:
        """
        The default text appended to any prompt to INDUCE CoT. 
        Implemented as separate property to facilitate CoT gap calculation.
        Can be overridden for customisation. 
        """
        return "Show your reasoning, step by step."
    

    @property
    def no_cot_instruction(self) -> str:
        """
        The default text appended to any prompt to SUPPRESS CoT. 
        Used in CoT gap calculation.
        Can be overridden for customisation. 
        """
        return "State your answer immediately, with no intervening steps."
     
     
    @abstractmethod
    def generate_prompt(self, index) -> str:
        """
        A user-implemented method to create the main body of LLM prompts for the task. 
        This method should return a REPRODUCIBLE, UNIQUE example of the task for each index, or None if no more examples are possible.

        EXAMPLES
        Example 1: for a dataset-based task, the index could refer to an example in the dataset.
        Method returns None if index > number of examples in set. 

        Example 2: for a game-based task, the index could be used as a random seed, which results in a unique game initialisation prompt. 
        
        NOTE: By default, self.cot_instruction will be appended to this in generate_full_prompt() to induce CoT. 
        NOTE: For dataset-based tasks, where this function can be called a limited number of times due to limited dataset size,
        this function MUST RETURN 'None' when the dataset is out of samples. 
        """
        pass


    def generate_full_prompt(self, index) -> str:
        """
        The API handle to get a new prompt. 
        Returns CoT/No-CoT pair for calculating CoT gap; discard No-CoT if not needed.  
        """
        main_body = self.generate_prompt(index)
        # Handle exit condition for limited-length dataset tasks
        if main_body is None:
            return None
        # Type check
        if not isinstance(main_body, str):
            raise TypeError(f'.generate_prompt() implementation must only return a string or None. Returned type: {type(main_body)}')
        cot_prompt = f'{main_body}\n\n{self.cot_instruction}'
        no_cot_prompt = f'{main_body}\n\n{self.no_cot_instruction}'
        return cot_prompt, no_cot_prompt

    
    @abstractmethod
    def get_task_score(
        self,
        prompt_data: PromptData,
    ) -> float:
        """
        TODO: update docstring

        A method to assess the success of the agent in the task, based on the Payload (prompt, additional info) and the final answer.
        - The cot is NOT included as an argument here, as the CoT is just a scratchpad for the agent.
        Only the final answer should be considered in assessing task performance.
        
        Score must be in range [0,1], and be higher for task success. 
        
        EXAMPLES
        If the task is a maths problem, this method might:
            (ii) Look up the right answer from a dataset held in memory
            (iii) Compare the two
            (iv) Emit 1 if correct or 0 otherwise. 
        Alternatively, if the task is a game, this method might:
            (i) Examine the prompt (game initialisation) and answers (moves/plays) and determine the game score for each player.
            (ii) Return a score between 0 and 1 representing success of agent in game. 
        """
        task_score = None
        return task_score


    def reward(self,
        task_score: float,
        language_score: float,
        t_weight: float = 1.0,
        l_weight: float = 1.0
    ) -> float:
        """
        Tasks which need a custom reward function can simply override this default   
        """

        # Reward for task success; penalise for use of language aspect        
        r = t_weight*task_score + l_weight*(1 - language_score)

        # Weighted score normalised to range [0,1]
        r /= (t_weight + l_weight)

        return r


    def reward_from_transcript(
        self,
        prompt_data: PromptData,
        t_weight: float = 1.0,
        l_weight: float = 1.0
    ) -> float:
        
        """
        The API handle to run user-set methods on all details from an episode

        Runs a check on output from score functions
        """


        task_score = self.get_task_score(prompt_data)
        self.check_score('task_score', task_score)
        
        language_score = self.language_aspect.get_language_score(prompt_data)
        self.check_score('language_score', language_score)

        r = self.reward(task_score, language_score, t_weight, l_weight)

        return r, task_score, language_score
    

    def check_score(self, score_name, score_value):
        if not isinstance(score_value, float):
            return TypeError(f'.get_{score_name}() must return a float. Returned type: {type(score_value)}')
        if score_value < 0 or score_value > 1: 
            return ValueError(f'.get_{score_name}() returned value {score_value}, which is outside range [0,1]')
    
    @abstractmethod
    def iterate_data(self, shuffle = True):
        raise NotImplementedError



class DatasetTask(Task):

    def __init__(self, dataset, language_aspect) -> None:
        super().__init__(language_aspect = language_aspect)
        self.dataset = pd.read_csv(dataset)
        self.length = len(self.dataset)
    
    def iterate_data(self, shuffle = True):
        """
        This assumes a dataset-like task
        This will need to be overwritten by game-like tasks
        """
        if shuffle:
            order_of_indices = torch.randperm(self.length)                # TODO: test reproducability of this shuffle
        else:
            order_of_indices = range(self.length)
        for idx in order_of_indices:
            cot_prompt, no_cot_prompt = self.generate_full_prompt(idx)
            yield PromptData(
                cot_prompt=cot_prompt,
                no_cot_prompt=no_cot_prompt,
                info=FixedDatasetDataInfo(idx = idx)
            )


class GameBasedTask(Task):
    """Each game will need to implement its own iterate_data method"""

    def __init__(self, language_aspect) -> None:
        super().__init__(language_aspect = language_aspect)


