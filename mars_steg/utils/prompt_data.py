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
    cot_prompt: str
    no_cot_prompt: str
    info: DataInfo
    cot_transcript: Optional[str] = None
    no_cot_transcript: Optional[str] = None

    def register_transcript(self, transcript: str, cot_mode: bool):
        if cot_mode:
            self.cot_transcript = transcript
        else:
            self.no_cot_transcript = transcript

