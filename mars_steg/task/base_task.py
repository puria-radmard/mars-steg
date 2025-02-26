"""
Module Name: base_task.py
------------------------

Description:
------------

This module contains the implementation of the base classes for tasks in the mars-steg framework to construct a task to train and evaluate an LLM for elicitng steganography.

Classes:
    - Task
    - TorchDatasetTask
    - GameBasedTask

Examples:
    Cf. mars_steg.README.md

"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
import torch
from torch.utils.data import Dataset

from mars_steg.utils.prompt_data import (
    FixedDatasetDataInfo,
    PromptData,
    BatchPromptData,
)
from mars_steg.model import BaseModel
from mars_steg.utils.score import check_score
from trl import PreTrainedModelWrapper
from transformers import AutoTokenizer

from mars_steg.task.neural_assessor import ProblemSolutionAssessor


import pandas as pd

from mars_steg.language.base_language_aspect import LanguageAspect
from mars_steg.config import PromptConfig

from typing import List, Tuple, Optional, Dict



class Task(metaclass=ABCMeta):
    """
    A base class for single-turn tasks.

    This abstract base class defines a structure for single-turn tasks that 
    require language processing. Subclasses should specify a task `name` and 
    ensure compatibility with a given `LanguageAspect`.

    Attributes
    ----------
    name : str
        The name of the task. Used to check compatibility with the `LanguageAspect`.
    system_prompt : str
        A system prompt specific to the task.
    uses_local_neural_assessor : bool
        Whether the task requires a local neural assessor for evaluation.
    language_aspect : LanguageAspect
        The language aspect configuration specifying compatibility with tasks.
    uses_local_neural_assessor : bool
        Indicates if a local neural assessor is needed for the task.
    prompt_config : PromptConfig
        Configuration object containing prompt templates.

    Methods
    -------
    cot_instruction() -> str
        Returns the default instruction appended to prompts to induce Chain-of-Thought (CoT).
    no_cot_instruction() -> str
        Returns the default instruction appended to prompts to suppress Chain-of-Thought (CoT).
    generate_prompt(index: int) -> Optional[str]
        Generates the main body of the LLM prompt for the task.
    generate_info(idx: int) -> FixedDatasetDataInfo
        Generates additional metadata or context for a task instance.
    generate_full_prompt(index: int) -> Optional[Tuple[str, str]]
        Generates a new task prompt, returning both CoT and No-CoT versions.
    get_task_score(prompt_data: PromptData, with_cot: bool) -> float
        Evaluates task performance based on PromptData object.
    reward(task_score: float, language_score: float, t_weight: float = 1.0, l_weight: float = 1.0) -> float
        Computes a weighted reward score based on task success and language aspect.
    reward_from_transcript(prompt_data: PromptData, t_weight: float = 1.0, l_weight: float = 1.0) -> Tuple[float, float, float]
        Computes the language score, the task score and calculates the final reward from a task transcript.
    recruit_neural_assessor(model: BaseModel)
        Initializes a neural assessor for evaluating problem/solution task type.


    Raises
    ------
    TypeError
        If the task name is not listed in `language_aspect.compatible_tasks`, 
        indicating an incompatibility issue.
    """



    name: str
    system_prompt: str
    uses_local_neural_assessor: bool

    def __init__(
        self, 
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
        prompt_config: PromptConfig,
    ):
        self.language_aspect = language_aspect
        self.uses_local_neural_assessor = uses_local_neural_assessor
        self.prompt_config = prompt_config  #TODO Store prompt configuration ?
        self.prompt_config = prompt_config  #TODO Store prompt configuration ?

        if self.name not in language_aspect.compatible_tasks:
            raise TypeError(f"""
                This task is not compatible with the submitted LanguageAspect instance.
                If there is no real compatibility issue, just update the compatible_tasks 
                property in your LanguageAspect. You should add this task's name property, 
                which is '{self.name}'.
                If there is no real compatibility issue, just update the compatible_tasks 
                property in your LanguageAspect. You should add this task's name property, 
                which is '{self.name}'.
            """)

    @property
    def cot_instruction(self) -> str:
        """
        Returns the default instruction appended to prompts to induce Chain-of-Thought (CoT).

        This property provides a standard instruction to encourage CoT reasoning in model responses. 
        It is implemented separately to facilitate CoT gap calculations and can be overridden for customization.

        Returns
        -------
        str
            The CoT-inducing instruction from the prompt configuration.
        """
        return self.prompt_config.with_cot_instruction

    @property
    def no_cot_instruction(self) -> str:
        """
        Returns the default instruction appended to prompts to suppress Chain-of-Thought (CoT).

        This property provides an instruction that explicitly prevents CoT reasoning in model responses. 
        It is used for CoT gap calculations and can be overridden for customization.

        Returns
        -------
        str
            The CoT-suppressing instruction from the prompt configuration.
        """
        return self.prompt_config.no_cot_instruction



    @abstractmethod
    def generate_prompt(self, index: int) -> Optional[str]:
        """
        Generates the main body of the LLM prompt for the task.

        This method must be implemented by subclasses to create a reproducible, 
        unique example of the task for a given index. If no more examples are available, 
        it should return `None`.

        Examples
        --------
        - **Dataset-based task**: The `index` corresponds to an example in a dataset. 
        If `index` exceeds the number of available examples, the method returns `None`.
        - **Game-based task**: The `index` serves as a random seed, producing a unique 
        game initialization prompt.

        Notes
        -----
        - By default, `self.cot_instruction` will be appended in `generate_full_prompt()` 
        to induce Chain-of-Thought (CoT).
        - If the number of available prompts is limited (e.g., in dataset-based tasks), 
        this method **must return `None`** when no more samples are available.

        Parameters
        ----------
        index : int
            An index specifying which example or scenario to generate.

        Returns
        -------
        Optional[str]
            The generated prompt text or `None` if no more examples are available.
        """
        pass


    @abstractmethod
    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        """
        Generates additional metadata or context for a task instance.

        This method must be implemented by subclasses to return a `FixedDatasetDataInfo` 
        object, which provides supplementary information related to the task at a given index.

        Parameters
        ----------
        idx : int
            The index specifying which task instance to retrieve additional information for.

        Returns
        -------
        FixedDatasetDataInfo
            An object containing metadata or extra context related to the specified task.
        Generates additional metadata or context for a task instance.

        This method must be implemented by subclasses to return a `FixedDatasetDataInfo` 
        object, which provides supplementary information related to the task at a given index.
        """
        pass


    def generate_full_prompt(self, index: int) -> Optional[Tuple[str, str]]:
        """
        Generates a new task prompt, returning both CoT and No-CoT versions.

        This method retrieves the main task prompt and appends instructions 
        for inducing or suppressing Chain-of-Thought (CoT) reasoning. 
        It facilitates CoT gap calculations by providing both variants.

        Parameters
        ----------
        index : int
            The index specifying which task prompt to generate.

        Returns
        -------
        Optional[Tuple[str, str]]
            A tuple containing:
            - `cot_prompt` (str): The task prompt with CoT instruction.
            - `no_cot_prompt` (str): The task prompt with No-CoT instruction.
            Returns `None` if no more examples are available.

        Raises
        ------
        TypeError
            If `generate_prompt()` does not return a string or `None`.

        Notes
        -----
        - If `generate_prompt()` returns `None`, this method also returns `None`.
        - This method is primarily used for CoT gap analysis; discard the No-CoT version if not needed.
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


        cot_prompt = f"{main_body}\n\n{self.cot_instruction}\n\n"
        no_cot_prompt = f"{main_body}\n\n{self.no_cot_instruction}\n\n"
        return cot_prompt, no_cot_prompt

    @abstractmethod
    def get_task_score(self, prompt_data: PromptData, with_cot: bool, skipping_failed_parsing_examples: bool = True) -> float:
        """
        Evaluates task performance based on PromptData object (cf. See Also).

        This method assesses whether the agent successfully completed the task 
        by analyzing the generated response in relation to the correct answer. 
        The Chain-of-Thought (CoT) reasoning process is **not** considered — only 
        the final answer affects the score.

        Parameters
        ----------
        prompt_data : PromptData
            The prompt and associated task information, including model-generated responses.
        with_cot : bool
            Whether the task was performed with CoT reasoning.
        skipping_failed_parsing_examples: bool 
            Skip examples that failed when parsing them for task score, language score, answer extraction, CoT extrcation etc

        Returns
        -------
        float
            A score in the range [0, 1], where a higher value indicates greater task success.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Examples
        --------
        **For a math problem**:
            - Look up the correct answer from a dataset.
            - Compare the model's final answer to the correct one.
            - Return `1.0` if correct, `0.0` otherwise.

        **For a game-based task**:
            - Analyze the game prompt and moves made by the agent.
            - Compute a success metric (e.g., win rate or score normalization).
            - Return a score between `0` and `1` reflecting performance.

        Notes
        -----
        - The method may use different evaluation strategies depending on the task type.
        - The neural assessor option must also be considered (cf. See Also).

        See Also
        --------
        mars_steg.utils.prompt_data.PromptData
            The structure of the PromptData object used in this method.
        MATH_Torch_Dataset_Task.get_task_score
            An example implementation for a dataset-based math task.  

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
        Computes the harmonic weighted reward score based on task success and language aspect.

        This method calculates a reward by combining the task performance score 
        with a penalty based on the language aspect. The final score is normalized 
        to the range [0, 1].

        Parameters
        ----------
        task_score : float
            A score in the range [0, 1] indicating task performance.
        language_score : float
            A score in the range [0, 1] indicating adherence to language constraints. It equals in our cases often to 1 - penalisation_score, where penalisation_score is included between 0 and 1.
        t_weight : float, optional
            The weight assigned to the task score (default is 1.0).
        l_weight : float, optional
            The weight assigned to the language score penalty (default is 1.0).

        Returns
        -------
        float
            A normalized reward score in the range [0, 1].

        Notes
        -----
        - First implementation was using a weighted average score, which induced the following issue : a random agent would have a constant reward of 0.5, whereas , in pricing game, a failing calculation of bid additioned to using P1 information was leading to a reward of 0. 
        In math task example, saying random things without using equal sign was leading as well to a reward 0.5, where trying to solve the problem was inducing a worst reward. We hope to fix this problematic case with a weighted harmonic reward.
        - A higher `task_score` increases the reward.
        - A higher `language_score` decreases the reward.
        - A low task score or language score induces a low reward.
        - Custom reward functions can be implemented by overriding this method.
        """
        r = (t_weight + l_weight) * task_score * language_score / (t_weight * language_score + l_weight * task_score + torch.finfo(torch.float32).eps)

        return r

    def reward_from_transcript(
        self, 
        prompt_data: PromptData, 
        t_weight: float = 1.0, 
        l_weight: float = 1.0,
        skipping_failed_parsing_examples: Optional[bool] = False
    ) -> Tuple[float, float, float]:
        """
        Computes the language score, the task score and calculates the final reward from a task transcript, i.e. the LLM answer for the considered task.

        This method evaluates both task and language scores using the 
        appropriate scoring functions, validates them, and computes 
        the final reward.

        Parameters
        ----------
        prompt_data : PromptData
            The transcript data containing details of the episode.
        t_weight : float, optional
            The weight assigned to the task score (default is 1.0).
        l_weight : float, optional
            The weight assigned to the language score (default is 1.0).
        XXX: skipping_failed_parsing_examples: bool, optional
            Skip examples that failed when parsing them for task score, language score, answer extraction, CoT extraction etc

        Returns
        -------
        Tuple[float, float, float]
            - `reward` (float): The final computed reward.
            - `task_score` (float): The evaluated task score.
            - `language_score` (float): The evaluated language aspect score.

        Notes
        -----
        - Assumes that the model was **prompted with Chain-of-Thought (CoT)**.
        - Calls `get_task_score()` to evaluate task performance.
        - Calls `get_language_score()` from `LanguageAspect` to assess language usage.
        - Uses `check_score()` to validate scores before computing reward.
        """
       
        task_score = self.get_task_score(prompt_data, with_cot=True)
        task_score = check_score('task_score:', task_score)
        
        language_score = self.language_aspect.get_language_score(prompt_data)
        language_score = check_score('language_score:', language_score)

        r = self.reward(task_score, language_score, t_weight, l_weight)

        return r, task_score, language_score

    def recruit_neural_assessor(self, model: BaseModel):
        """
        Initializes a neural assessor for evaluating problem/solution task type.

        This method assigns a neural assessment model to the task, enabling 
        automated evaluation of generated solutions.

        Parameters
        ----------
        model : BaseModel
            The neural model to be used for solution assessment (cf. See Also).

        Raises
        ------
        AssertionError
            If `uses_local_neural_assessor` is `False`, indicating that 
            the task does not support neural assessment.

        Notes
        -----
        - This method should only be called if `uses_local_neural_assessor` is `True`.
        - The recruited model is assigned to `self.whitebox_model`.
        - Initializes a `ProblemSolutionAssessor` with the given model.

        See Also
        --------
        mars_steg.task.neural_assessor.ProblemSolutionAssessor
            The neural assessor class used for evaluating problem/solution tasks.
        mars_steg.model.base_model.py
            The base model class used for neural models.
        
        """
        assert self.uses_local_neural_assessor, (
            f"Attempting to recruit neural assessor to a {self.__class__.__name__}, "
            "which does not use it!"
        )
       
        self.whitebox_model = model
        self.neural_assessor = ProblemSolutionAssessor(
            model=model, prompt_config=self.prompt_config
        )





class TorchDatasetTask(Task, Dataset):
    """
    A PyTorch dataset-based task that integrates with prompt-based LLM evaluations.

    This class extends `Task` and `Dataset`, handling structured dataset-based 
    tasks where prompts are derived from dataset entries. It also supports 
    neural assessment for evaluating model-generated solutions.

    Attributes
    ----------
    dataset : str
        Path to the dataset file (CSV format).
    language_aspect : LanguageAspect
        The language aspect settings for this task.
    uses_local_neural_assessor : bool
        Whether the task utilizes a local neural assessor for evaluation.
    prompt_config : PromptConfig
        The prompt configuration used for generating prompts.
    batch_size : int
        Batch size for torch dataloaders

    Methods
    -------
    generate_info(idx: int) -> FixedDatasetDataInfo
        Generates a fixed dataset information object for a given index.
    test_train_split(train_proportion: float, validation_proportion: float) -> Tuple[TorchDatasetTask, TorchDatasetTask, TorchDatasetTask]
        Splits the dataset into train, validation, and test sets.
    prompt_data_collate_fn(batch: List[PromptData]) -> BatchPromptData
        Aggregates a batch of `PromptData` into a `BatchPromptData` object.
    get_true_answers_from_batch_prompt_data(prompt_data: BatchPromptData) -> List[str]
        Extracts the ground-truth answers for a batch of prompts.
    neural_assessor_from_batch_prompt_data(prompt_data: BatchPromptData, from_cot_prompt: bool) -> BatchPromptData
        Generates neural assessor responses for a batch of prompts.


    Notes
    -----
    This class initializes several attributes to None that can be set later using recruit_neural_assessor method.
    """

    def __init__(self,
        dataset: str,
        language_aspect: LanguageAspect,
        uses_local_neural_assessor: bool,
        prompt_config: PromptConfig,
        batch_size: int
    ) -> None:
        super().__init__(
            language_aspect=language_aspect,
            uses_local_neural_assessor=uses_local_neural_assessor,
            prompt_config=prompt_config
        )
        self.dataset_name = dataset
        self.dataset = pd.read_csv(dataset)
        self.length = len(self.dataset)
        self.prompt_config = prompt_config
        self.batch_size = batch_size

        if self.uses_local_neural_assessor:
            self.whitebox_model: Optional[PreTrainedModelWrapper] = None
            self.tokenizer: Optional[AutoTokenizer] = None
            self.generation_kwargs: Optional[Dict] = None
            self.neural_assessor: Optional[ProblemSolutionAssessor] = None

    def generate_info(self, idx: int) -> FixedDatasetDataInfo:
        """
        Generates a fixed dataset information object for a given index.

        This method creates a `FixedDatasetDataInfo` object containing metadata
        or context related to the dataset at the specified index.

        Parameters
        ----------
        idx : int
            The index of the dataset sample.

        Returns
        -------
        FixedDatasetDataInfo
            A structured information object for the dataset sample.
        """
        return FixedDatasetDataInfo(idx=idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves a dataset sample and generates corresponding CoT and No-CoT prompts.

        This method assumes a dataset-like task structure and must be overridden 
        for game-like tasks.

        Parameters
        ----------
        idx : int
            The index of the dataset sample.

        Returns
        -------
        PromptData
            A structured prompt containing both CoT and No-CoT versions.

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
        """
        Splits the dataset into train, validation, and test sets.

        The method partitions the dataset based on the specified proportions, ensuring 
        that the total size does not exceed the dataset length.

        Parameters
        ----------
        train_proportion : float
            The proportion of the dataset to allocate for training (0 to 1).
        validation_proportion : float
            The proportion of the dataset to allocate for validation (0 to 1).

        Returns
        -------
        Tuple[TorchDatasetTask, TorchDatasetTask, TorchDatasetTask]
            A tuple containing the train, validation, and test dataset tasks.

        Raises
        ------
        AssertionError
            If proportions are out of range or exceed the dataset size.

        Notes
        -----
        - The test set is automatically determined as the remaining portion.
        - The dataset is not shuffled before splitting (TODO: Implement shuffling).
        """

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
        # TODO: shuffled selection
        train_dataset_pd = self.dataset.iloc[:train_size]
        val_dataset_pd = self.dataset.iloc[train_size:train_size + validation_size]
        test_dataset_pd = self.dataset.iloc[train_size + validation_size:]

        print(f"Train size: {len(train_dataset_pd)}")
        print(f"Validation size: {len(val_dataset_pd)}")
        print(f"Test size: {len(test_dataset_pd)}")
        
        cls = self.__class__
        output_datasets = []
        for dataset_pd in [train_dataset_pd, val_dataset_pd, test_dataset_pd]:
            new_dataset = cls(self.dataset_name, self.language_aspect, self.uses_local_neural_assessor, self.prompt_config, self.batch_size)
            new_dataset.dataset = dataset_pd
            new_dataset.length = len(dataset_pd)
            if self.uses_local_neural_assessor:
                new_dataset.recruit_neural_assessor(
                    model=self.whitebox_model
                )
            output_datasets.append(new_dataset)

        return output_datasets

    def prompt_data_collate_fn(self, batch: List[PromptData]) -> BatchPromptData:
        """
        Aggregates a batch of `PromptData` into a `BatchPromptData` object.

        This method is useful for preparing batch inputs for model inference.

        Parameters
        ----------
        batch : List[PromptData]
            A list of individual `PromptData` objects.

        Returns
        -------
        BatchPromptData
            A batched structure containing all prompt data.
        """


        return BatchPromptData(prompt_datas=batch)

    def get_true_answers_from_batch_prompt_data(self, prompt_data: BatchPromptData) -> List[str]:
        """
        Extracts the ground-truth answers for a batch of prompts.

        Parameters
        ----------
        prompt_data : BatchPromptData
            A batch of PromptData (cf . See Also).

        Returns
        -------
        List[str]
            A list of true answers corresponding to each prompt in the batch.

        See Also
        --------
        mars_steg.utils.prompt_data.PromptData
            The structure of the PromptData object used in this method.
        """

        idxs = [pii.idx for pii in prompt_data.infos]
        true_answers = [self.dataset.iloc[idx]["True answer"] for idx in idxs]
        return true_answers

    def neural_assessor_from_batch_prompt_data(
        self, prompt_data: BatchPromptData, from_cot_prompt: bool, neural_assessor_thinking_helper: str = None
    ) -> BatchPromptData:
        """
        Generates neural assessor responses for a batch of prompts.

        This method utilizes the neural assessor to evaluate model-generated 
        answers and extract structured assessment data.

        Parameters
        ----------
        prompt_data : BatchPromptData
            The batch of PromptData objects to assess.
        from_cot_prompt : bool
            Whether to the BatchPromptData contains CoT prompts.

        Returns
        -------
        BatchPromptData
            A copy of the initial BatchPromptData containing the generated assessor responses.

        Raises
        ------
        AssertionError
            If `uses_local_neural_assessor` is `False` or `neural_assessor` is not initialized.
        """

        assert self.uses_local_neural_assessor and self.neural_assessor is not None
        true_answers = self.get_true_answers_from_batch_prompt_data(prompt_data)
        return self.neural_assessor.get_assessor_generated_answers(prompt_data, true_answers, from_cot_prompt, neural_assessor_thinking_helper)


class GameBasedTask(Task):
    """
    A base class for game-based tasks.

    This abstract base class defines a structure for game-based tasks (cf. Task).

    See Also
    --------

    Task
        The base class for single-turn tasks.

    """

    def __init__(self, language_aspect) -> None:
        super().__init__(language_aspect=language_aspect)
