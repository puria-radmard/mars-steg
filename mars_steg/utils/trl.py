from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import yaml


@dataclass
class GenerationKwargs:
    min_length: int
    top_k: int
    top_p: int
    num_beams: int
    temperature: float
    do_sample: bool
    pad_token_id: int
    max_new_tokens: int
    return_prompt: bool
    generate_ref_response: bool
    length_sampler: Optional[Callable] = None

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
