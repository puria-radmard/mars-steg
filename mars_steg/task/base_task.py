from __future__ import annotations

from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import Dataset

from mars_steg.utils.prompt_data import (
    FixedDatasetDataInfo,
    PromptData,
    BatchPromptData,
)
from mars_steg.utils.score import check_score


import pandas as pd

from mars_steg.language.base_language_aspect import LanguageAspect


from typing import List, Tuple



class Task(metaclass=ABCMeta):
    """
    A base class for single-turn tasks
    """

    """
    name property: used to check submitted language property is compatible with this task

    Set as the name of your subclass by default. 
    """
    name: str
    system_prompt: str

    def __init__(self, language_aspect: LanguageAspect, use_neural_assessor: bool):
        self.language_aspect = language_aspect
        self.use_neural_assessor = use_neural_assessor
        if self.name not in language_aspect.compatible_tasks:
            raise TypeError(f"""
                This task is not compatible with the submitted LanguageAspect instance.
                If there is no real compatibility issue, just update the compatible_tasks property in your LanguageAspect.
                You should add this tasks' name property, which is '{self.name}'.
            """)

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
    
    @abstractmethod
    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        """
        A method to generate additional information for the task at the given index.
        This method should return a FixedDatasetDataInfo object, which can be used to store additional information about the task.
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
            raise TypeError(
                f".generate_prompt() implementation must only return a string or None. Returned type: {type(main_body)}"
            )
        cot_prompt = f"{main_body}\n\n{self.cot_instruction}"
        no_cot_prompt = f"{main_body}\n\n{self.no_cot_instruction}"
        return cot_prompt, no_cot_prompt

    @abstractmethod
    def get_task_score(self, prompt_data: PromptData, with_cot: bool) -> float:
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
        raise NotImplementedError

    def reward(
        self,
        task_score: float,
        language_score: float,
        t_weight: float = 1.0,
        l_weight: float = 1.0,
    ) -> float:
        """
        Tasks which need a custom reward function can simply override this default
        """

        # Reward for task success; penalise for use of language aspect
        r = t_weight * task_score + l_weight * (1 - language_score)

        # Weighted score normalised to range [0,1]
        r /= t_weight + l_weight

        return r

    def reward_from_transcript(
        self, prompt_data: PromptData, t_weight: float = 1.0, l_weight: float = 1.0
    ) -> float:
        """
        The API handle to run user-set methods on all details from an episode

        Runs a check on output from score functions

        Full thing assumes we prompted with cot
        """

        task_score = self.get_task_score(prompt_data, with_cot = True)
        task_score = check_score('task_score', task_score)
        
        language_score = self.language_aspect.get_language_score(prompt_data)
        language_score = check_score('language_score', language_score)

        r = self.reward(task_score, language_score, t_weight, l_weight)

        return r, task_score, language_score



class TorchDatasetTask(Task, Dataset):

    def __init__(self, dataset: str, language_aspect: LanguageAspect, use_neural_assessor: bool) -> None:
        super().__init__(language_aspect=language_aspect, use_neural_assessor=use_neural_assessor)
        self.dataset_name = dataset
        self.dataset = pd.read_csv(dataset)
        self.length = len(self.dataset)

    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        return FixedDatasetDataInfo(idx=idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        This assumes a dataset-like task
        This will need to be overwritten by game-like tasks
        """

        cot_prompt, no_cot_prompt = self.generate_full_prompt(idx)
        return PromptData(
            cot_prompt=cot_prompt,
            no_cot_prompt=no_cot_prompt,
            info=self.generate_info(idx),
        )

    def test_train_split(
        self, train_proportion: float, validation_proportion: float
    ) -> Tuple[TorchDatasetTask, TorchDatasetTask, TorchDatasetTask]:
        assert 0 <= train_proportion <= 1, "train_proportion must be between 0 and 1"
        assert 0 <= validation_proportion <= 1, "validation_proportion must be between 0 and 1"
        assert train_proportion + validation_proportion <= 1, (
        "train_proportion + validation_proportion must be <= 1"
    )

        dataset_length = len(self.dataset)  # Ensure correct dataset length
        train_size = int(dataset_length * train_proportion)
        validation_size = int(dataset_length * validation_proportion)
        test_size = dataset_length - train_size - validation_size

        # Ensure sizes do not exceed the dataset length
        assert train_size + validation_size + test_size == dataset_length, (
            "Train, validation, and test sizes must add up to the dataset length"
        )

        # Slicing the dataset
        train_dataset_pd = self.dataset.iloc[:train_size]
        val_dataset_pd = self.dataset.iloc[train_size:train_size + validation_size]
        test_dataset_pd = self.dataset.iloc[train_size + validation_size:]

        print(f"Train size: {len(train_dataset_pd)}")
        print(f"Validation size: {len(val_dataset_pd)}")
        print(f"Test size: {len(test_dataset_pd)}")
        cls = self.__class__
        train_dataset = cls(self.dataset_name, self.language_aspect)
        train_dataset.dataset = train_dataset_pd
        train_dataset.length = len(train_dataset_pd)
        val_dataset = cls(self.dataset_name, self.language_aspect)
        val_dataset.dataset = val_dataset_pd
        val_dataset.length = len(val_dataset_pd)
        test_dataset = cls(self.dataset_name, self.language_aspect)
        test_dataset.dataset = test_dataset_pd
        test_dataset.length = len(test_dataset_pd)
        return train_dataset, val_dataset, test_dataset

    def iterate_data(self, shuffle=True):
        """
        Will need to be modified at the end of the branch completed
        """
        pass

    def prompt_data_collate_fn(self, batch: List[PromptData]) -> BatchPromptData:

        return BatchPromptData(prompt_datas=batch)


class GameBasedTask(Task):
    """Each game will need to implement its own iterate_data method"""

    def __init__(self, language_aspect) -> None:
        super().__init__(language_aspect=language_aspect)
