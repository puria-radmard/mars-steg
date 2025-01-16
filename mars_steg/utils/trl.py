from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import yaml


@dataclass
class GenerationKwargs:
    min_length: int
    num_beams: int
    do_sample: bool
    pad_token_id: int
    max_new_tokens: int
    return_prompt: bool
    generate_ref_response: bool
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    length_sampler: Optional[Callable] = None

    def __post_init__(self):
        if not self.do_sample:
            assert self.temperature == self.top_k == self.top_p == None,\
                "Cannot specify GenerationKwargs.temperature if GenerationKwargs.do_sample is specified"

    @classmethod
    def from_yaml(cls, yaml_file):
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def to_yaml(self, yaml_file):
        with open(yaml_file, "w") as file:
            yaml.dump(asdict(self), file)
