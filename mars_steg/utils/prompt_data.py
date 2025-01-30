from abc import ABC
from typing import List, Optional, Iterable
from dataclasses import dataclass, asdict
from datetime import datetime

from torch import Tensor as _T

import wandb


@dataclass
class DataInfo(ABC):

    """
    Base class for the customisable component of PromptData (PromptData.info)
    """

    pass


@dataclass
class FixedDatasetDataInfo(DataInfo):

    """
    DataInfo subclass suitable for dataset-based tasks
    """

    idx: int


@dataclass
class SequentialPriceTaskDataInfo(FixedDatasetDataInfo):

    """
    DataInfo subclass suitable for the sequential price task
    """

    p1_bid: float
    p2_true_cost: float


@dataclass
class PromptData:

    """
    PromptData is an object designed to carry ALL data associated with a single episode of RL training.

    It starts with just cot_prompt, no_cot_prompt and a customisable 'info' attribute which can carry additional task-specific information.

    As an iteration of the training loop progresses, more data is added to the optional attributes until this object contains all data associated with a single episode. 
    """

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

    overseer_prompt: Optional[str] = None
    assessor_prompt: Optional[str] = None
    
    overseer_transcript: Optional[str] = None # Full transcript from the overseer
    extracted_overseer_answer: Optional[bool] = None # Final decision from overseer
    assessor_transcript: Optional[str] = None # Full transcript from the assessor
    extracted_assessor_answer: Optional[bool] = None # Final decision from overseer

    def __str__(self) -> str:
        return (
            f"cot_prompt: {self.cot_prompt}\n"
            f"no_cot_prompt: {self.no_cot_prompt}\n"
            f"info: {self.info}\n"
            f"cot_transcript: {self.cot_transcript}\n"
            f"no_cot_transcript: {self.no_cot_transcript}\n"
            f"overseer_transcript: {self.overseer_transcript}\n"
            f"extracted_cot: {self.extracted_cot}\n"
            f"extracted_final_answer_with_cot: {self.extracted_final_answer_with_cot}\n"
            f"extracted_final_answer_without_cot: {self.extracted_final_answer_without_cot}\n"
            f"overseer_transcript: {self.overseer_transcript}\n"
            f"extracted_overseer_answer: {self.extracted_overseer_answer}\n"
            f"assessor_transcript: {self.assessor_transcript}\n"
            f"extracted_assessor_answer: {self.extracted_assessor_answer}\n"
        )





class BatchPromptData:

    """
    BatchPromptData contains and manages a list of PromptData objects; see PromptData.
    """

    def __init__(self, prompt_datas: List[PromptData]):
        # assert len(prompt_datas) > 0, "BatchPromptData must have at least one PromptData"

        self.prompt_datas: List[PromptData] = prompt_datas
        self.cot_prompts: Optional[List[str]] 
        self.no_cot_prompts: Optional[List[str]] 
        self.infos: Optional[List[str]] 

        self.cot_transcripts: Optional[List[str]]
        self.no_cot_transcripts: Optional[List[str]]
        self.overseer_transcript: Optional[List[str]]

        self.extracted_cots: Optional[List[str]] 
        self.extracted_final_answer_with_cots: Optional[List[str]] 
        self.extracted_final_answer_without_cots: Optional[List[str]] 
        
        self.overseer_prompts: Optional[List[str]]
        self.assessor_prompts: Optional[List[str]]

        self.overseer_transcripts: Optional[List[str]] # Full transcript from the overseer
        self.extracted_overseer_answers: Optional[List[bool]] # Final decision from overseer
        self.assessor_transcripts: Optional[List[str]] # Full transcript from the assessor
        self.extracted_assessor_answers: Optional[List[bool]] # Final decision from overseer

    def __len__(self):
        return len(self.prompt_datas)

    def __iter__(self) -> Iterable[PromptData]:
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
        
    def __str__(self):
        for prompt_data in self.prompt_datas:
            print("------------------------------------")
            print(prompt_data)
            print("------------------------------------")
        return "\n"

    def log_to_wandb(
        self,
    ):
        data = []
        members_as_dicts = [asdict(pd) for pd in self.prompt_datas]
        columns = list(members_as_dicts[0].keys())
        for pd_dict in members_as_dicts:
            data.append(list(pd_dict.values()))
        wandb.log({f"Batch_prompt_data_{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}": 
                   wandb.Table(columns=columns, data=data)})


    
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
