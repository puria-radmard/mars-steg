from abc import ABC
from typing import List, Optional
from dataclasses import dataclass

from torch import Tensor as _T


@dataclass
class DataInfo(ABC):
    pass


@dataclass
class FixedDatasetDataInfo(DataInfo):
    idx: int


@dataclass
class SequentialPriceTaskDataInfo(FixedDatasetDataInfo):
    p1_bid: float
    p2_true_cost: float


@dataclass
class PromptData:

    # Ab initio data
    cot_prompt: str
    no_cot_prompt: str
    info: DataInfo

    # After model gives response
    cot_transcript: Optional[str] = None
    no_cot_transcript: Optional[str] = None

    # After (TODO) separates answers into cot and response
    extracted_cot: Optional[str] = None
    extracted_final_answer_with_cot: Optional[str] = None
    extracted_final_answer_without_cot: Optional[str] = None

    # TODO --> Puria after wandb
    def save(
        self,
    ):
        raise NotImplementedError


@dataclass
class BatchPromptData:

    prompt_datas: List[PromptData]

    def __len__(self):
        return len(self.prompt_datas)

    def __iter__(self):
        """Make the object iterable by returning an iterator over the prompt_datas list."""
        return iter(self.prompt_datas)

    def __getitem__(self, idx):
        return self.prompt_datas[idx]

    def __setitem__(self, idx, value):
        self.prompt_datas[idx] = value

    def __delitem__(self, idx):
        del self.prompt_datas[idx]

    def append(self, prompt_data: PromptData):
        self.prompt_datas.append(prompt_data)

    def index(self, prompt_data: PromptData):
        return self.prompt_datas.index(prompt_data)

    @property
    def cot_prompts(self) -> List[str]:
        return [prompt_data.cot_prompt for prompt_data in self.prompt_datas]

    @property
    def no_cot_prompts(self) -> List[str]:
        return [prompt_data.no_cot_prompt for prompt_data in self.prompt_datas]

    @property
    def infos(self) -> List[DataInfo]:
        return [prompt_data.info for prompt_data in self.prompt_datas]

    @property
    def cot_transcripts(self) -> List[Optional[str]]:
        return [prompt_data.cot_transcript for prompt_data in self.prompt_datas]

    @cot_transcripts.setter
    def cot_transcripts(self, transcripts: List[str]):
        for prompt_data, transcript in zip(self.prompt_datas, transcripts):
            prompt_data.cot_transcript = transcript

    @property
    def extracted_cots(self) -> List[Optional[str]]:
        return [prompt_data.extracted_cot for prompt_data in self.prompt_datas]

    @extracted_cots.setter
    def extracted_cots(self, extracted_cots: List[str]):
        for prompt_data, extracted_cot in zip(self.prompt_datas, extracted_cots):
            prompt_data.extracted_cot = extracted_cot

    @property
    def extracted_final_answers_with_cot(self) -> List[Optional[str]]:
        return [
            prompt_data.extracted_final_answer_with_cot
            for prompt_data in self.prompt_datas
        ]

    @extracted_final_answers_with_cot.setter
    def extracted_final_answers_with_cot(
        self, extracted_final_answers_with_cot: List[str]
    ):
        for prompt_data, extracted_final_answer_with_cot in zip(
            self.prompt_datas, extracted_final_answers_with_cot
        ):
            prompt_data.extracted_final_answer_with_cot = (
                extracted_final_answer_with_cot
            )

    @property
    def extracted_final_answers_without_cot(self) -> List[Optional[str]]:
        return [
            prompt_data.extracted_final_answer_without_cot
            for prompt_data in self.prompt_datas
        ]

    @extracted_final_answers_without_cot.setter
    def extracted_final_answers_without_cot(
        self, extracted_final_answers_without_cot: List[str]
    ):
        for prompt_data, extracted_final_answer_without_cot in zip(
            self.prompt_datas, extracted_final_answers_without_cot
        ):
            prompt_data.extracted_final_answer_without_cot = (
                extracted_final_answer_without_cot
            )

    # TODO --> Puria after wandb
    def save(
        self,
    ):
        raise NotImplementedError
