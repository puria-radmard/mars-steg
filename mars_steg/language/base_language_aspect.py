from typing import Set

from mars_steg.utils.prompt_data import PromptData
from mars_steg.model.base_model import BaseModel

from abc import ABCMeta, abstractmethod
from typing import Dict

from trl import PreTrainedModelWrapper
from transformers import AutoTokenizer


class LanguageAspect(metaclass=ABCMeta):
    """
    A base class for language aspects to penalise.

    Has been separated from Task to allow us to easily plug and play multiple LanguageAspects with the same Task.

    Requires you to implement:
    1. compatible_tasks: a property defining the set of Task subclasses this LanguageAspect is compatible with.
    Should include the 'name' property of each Task subclass you intend to use this LanguageAspect with.

    2. get_language_score: see the description below.
    """

    compatible_tasks: Set[str]
    
    # False in most cases except for the one case - but will be checked externally so needs to be defined everywhere
    uses_local_neural_overseer: bool

    @abstractmethod
    def get_language_score(
        self,
        prompt_data: PromptData,
    ) -> float:
        """
        A method to examine the CoT for examples of the language aspect we are intending to penalise.

        Score must be in range [0,1], and be zero if the language aspect is not detected, or higher if it is.

        EXAMPLES
        If the task is a maths problem, this method might:
            (i) Find the number of times the '=' sign is used in the cot
            (ii) Return a score between 0 and 1 which is higher for more '=' signs
        Alternatively, if the task was Mustafa's market game, this method might:
            (i) Send the cot to an LLM overseer with a prompt asking whether the agent CoT contains a reference to P1's prices
            (ii) Return 1 if it does, and 0 if it doesn't.
        """
        language_score = None
        return language_score

    def recruit_neural_overseer(self, reference_model: BaseModel, tokenizer: AutoTokenizer, generation_kwargs: Dict):
        assert self.uses_local_neural_overseer, f"Attempting to recruit neural overseer to a {self.__class__.__name__}, which does not use it!"
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.generation_kwargs = generation_kwargs


class EmptyLanguageAspect(LanguageAspect):

    uses_local_neural_overseer = False

    def __init__(self):
        super().__init__()

    def get_language_score(self, prompt_data: PromptData) -> float:
        return 0.0

