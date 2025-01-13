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
class SequentialPriceGameDataInfo(DataInfo):
    p1_bid: _T
    p2_true_cost: _T
    raw_material_cost: _T
    base_labor_hours: _T
    defect_rate: _T
    efficiency_factor: _T
    annual_volume: _T
    inventory_days: _T
    labor_rate: _T
    machine_cost: _T
    machine_life_years: _T
    daily_storage_cost: _T
    


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
