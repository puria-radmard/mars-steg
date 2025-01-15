from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import yaml


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./experiment",
        metadata={"help": "the path to save the model"},
    )
    seed: Optional[int] = field(default=42, metadata={"help": "the seed for reproducibility"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    num_eval_episodes: Optional[int] = field(default=10, metadata={"help": "the number of evaluation episodes"})



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
    batch_size: int
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
    

