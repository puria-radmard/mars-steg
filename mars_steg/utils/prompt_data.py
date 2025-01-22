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
    overseer_response: Optional[str] = None

    # After (TODO) separates answers into cot and response
    extracted_cot: Optional[str] = None
    extracted_final_answer_with_cot: Optional[str] = None
    extracted_final_answer_without_cot: Optional[str] = None
    extracted_overseer_response: Optional[bool] = None # Overseer shall answer True or False

    # TODO --> Puria after wandb
    def save(
        self,
    ):
        raise NotImplementedError


class BatchPromptData:

    def __init__(self, prompt_datas: List[PromptData]):
        assert len(prompt_datas) > 0, "BatchPromptData must have at least one PromptData"
        self.prompt_datas : List[PromptData] = prompt_datas
        self.cot_prompts: Optional[List[str]] 
        self.no_cot_prompts: Optional[List[str]] 
        self.infos: Optional[List[str]] 

        self.cot_transcripts: Optional[List[str]]
        self.no_cot_transcripts: Optional[List[str]]
        self.overseer_response: Optional[List[str]]

        self.extracted_cots: Optional[List[str]] 
        self.extracted_final_answer_with_cots: Optional[List[str]] 
        self.extracted_final_answer_without_cots: Optional[List[str]] 
        self.extracted_overseer_responses: Optional[List[bool]] # Overseer shall answer True or False‡


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
    
    def __setattr__(self, name, value):
        if name == "prompt_datas":
            super().__setattr__(name, value)
        else:
            # Normal use: Update attributes for all `prompt_datas` elements
            for i, prompt_data in enumerate(self.prompt_datas):
                setattr(prompt_data, name[:-1], value[i])
    
    def __getattr__(self, name):
        
        if name == "prompt_datas":
            return super().__getattr__(name)
        else:
            return [getattr(prompt_data, name[:-1]) for prompt_data in self.prompt_datas]

    # TODO --> Puria after wandb
    def save(
        self,
    ):
        raise NotImplementedError
    
if __name__ == "__main__":
    data_infos = [FixedDatasetDataInfo(idx=i) for i in range(10)]
    prompt_datas = [PromptData(cot_prompt=f"cot{data_info.idx}", no_cot_prompt=f"no cot{data_info.idx}", info=data_info) for data_info in data_infos]
    batch_prompt_data = BatchPromptData(prompt_datas)
    ex_1 = batch_prompt_data.prompt_datas
    print(ex_1)
    print(batch_prompt_data.cot_prompts)
    print(batch_prompt_data.no_cot_prompts)
    print(batch_prompt_data.infos)
    assert batch_prompt_data.cot_prompts == [f"cot{data_info.idx}" for data_info in data_infos]
    batch_prompt_data.cot_prompts = ["new cot"] * 10
    print(batch_prompt_data.cot_prompts)
    for prompt_data in batch_prompt_data:
        print(prompt_data.cot_prompt)
    
    batch_prompt_data.extracted_cots = ["extracted_cot"] * 10
    print(batch_prompt_data.extracted_cots)
    for prompt_data in batch_prompt_data:
        print(prompt_data.extracted_cot)
