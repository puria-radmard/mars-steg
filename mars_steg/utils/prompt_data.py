from abc import ABC
from typing import Optional
from dataclasses import dataclass

from torch import Tensor as _T


@dataclass
class DataInfo(ABC):
    pass


@dataclass
class FixedDatasetDataInfo(DataInfo):
    idx: int


@dataclass
class GameDataInfo(DataInfo):
    price: _T


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
    cot: Optional[str] = None
    final_answer: Optional[str] = None

    def register_transcript(self, transcript: str, cot_mode: bool):
        if cot_mode:
            self.cot_transcript = transcript
        else:
            self.no_cot_transcript = transcript

    def register_response_extraction(self, extraction: str, cot_mode: bool):
        if cot_mode:
            self.cot = extraction
        else:
            self.final_answer = extraction
