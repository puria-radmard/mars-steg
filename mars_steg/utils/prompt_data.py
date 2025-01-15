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
    p1_bid: float
    p2_true_cost: float
    raw_material_cost: float
    base_labor_hours: float
    defect_rate: float
    efficiency_factor: float
    annual_volume: float
    inventory_days: float
    labor_rate: float
    machine_cost: float
    machine_life_years: float
    daily_storage_cost: float
    penalise_in_prompt: bool
    profit_maximisation: bool
    


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
    def save(self,):
        raise NotImplementedError
