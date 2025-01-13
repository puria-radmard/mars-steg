from abc import ABCMeta, abstractmethod
import torch


class Task(metaclass=ABCMeta):

    """
    A base class for single-turn tasks
    """

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
     
     
    def iterator(self, shuffle=False):
        """
        Universal iterator with shuffle option for dataset-based (finite, offline prompt generation) 
        or game-based (infinite, online prompt generation) tasks.
        """
        def get_info(index):
            """Helper function to get base info dict and any additional info from child class"""
            base_info = {'index': index}
            # If child class has defined get_additional_info, call it
            if hasattr(self, 'get_additional_info'):
                additional_info = self.get_additional_info(index)
                base_info.update(additional_info)
            return base_info

        if hasattr(self, 'dataset_length'):  # Dataset-based task
            order = torch.randperm(self.dataset_length) if shuffle else range(self.dataset_length)
            
            for index in order:
                yield {
                    'cot_prompt': self.generate_full_prompt(index)[0],
                    'no_cot_prompt': self.generate_full_prompt(index)[1],
                    'info': get_info(index)
                }
        else:  # Game-based task
            index = 0
            while True:
                yield {
                    'cot_prompt': self.generate_full_prompt(index)[0],
                    'no_cot_prompt': self.generate_full_prompt(index)[1],
                    'info': get_info(index)
                }
                index += 1

    @abstractmethod
    def generate_prompt(self, index) -> str:
        """
        A user-implemented method to create the main body of LLM prompts for the task. 
        This method should return a REPRODUCIBLE, UNIQUE example of the task for each index, or None if no more examples are possible.

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
    def get_task_score(self, transcript: str) -> float:
        """
        A method to assess the success of the agent in the task, based on a complete transcript of the episode. 
        Score must be in range [0,1], and be higher for task success. 
        For example, if the task is a maths problem, this method might:
            (i) Extract the final answer from the transcript (perhaps using a reusable helper function)
            (ii) Look up the right answer from a dataset held in memory
            (iii) Compare the two
            (iv) Emit 1 if correct or 0 otherwise. 
        Alternatively, if the task is a game, this method might:
            (i) Examine the transcript and determine the game score for each player.
            (ii) Return a score between 0 and 1 representing success of agent in game. 
        """
        task_score = None
        return task_score


    @abstractmethod
    def get_language_score(self, transcript: str) -> float:
        """
        A method to examine the CoT for examples of the language aspect we are intending to penalise.
        Score must be in range [0,1], and be zero if the language aspect is not detected, or higher if it is. 
        For example, if the task is a maths problem, this method might:
            (i) Extract the CoT from the transcript (perhaps using a reusable helper function)
            (ii) Find the number of times the '=' sign is used
            (iii) Return a score between 0 and 1 which is higher for more '=' signs
        Alternatively, if the task was Mustafa's market game, this method might:
            (i) Extract a relevant part of the transcript
            (ii) Send it to an LLM overseer with a prompt asking whether the agent CoT contains a reference to P1's prices
            (iii) Return 1 if it does, and 0 if it doesn't.
        """
        language_score = None
        return language_score


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
        transcript: str,
        t_weight: float = 1.0,
        l_weight: float = 1.0
    ) -> float:
        
        """
        The API handle to run user-set methods on a transcript to determine episode reward

        Runs a check on output from score functions
        """

        task_score = self.get_task_score(transcript)
        self.check_score('task_score', task_score)
        
        language_score = self.get_language_score(transcript)
        self.check_score('language_score', language_score)

        r = self.reward(task_score, language_score, t_weight, l_weight)

        return r, task_score, language_score
    

    def check_score(self, score_name, score_value):
        if not isinstance(score_value, float):
            return TypeError(f'.get_{score_name}() must return a float. Returned type: {type(score_value)}')
        if score_value < 0 or score_value > 1: 
            return ValueError(f'.get_{score_name}() returned value {score_value}, which is outside range [0,1]')