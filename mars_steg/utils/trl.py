from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import yaml


@dataclass
class ScriptArguments:
    """
    The name of the Causal LM model we wish to fine-tune with PPO
    """
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "The model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "Use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "The learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "The PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "The batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./experiment",
        metadata={"help": "The path to save the model"},
    )
    seed: Optional[int] = field(default=42, metadata={"help": "The seed for reproducibility"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "The number of training epochs"})
    num_eval_episodes: Optional[int] = field(default=10, metadata={"help": "The number of evaluation episodes"})
    save_frequency: Optional[int] = field(default=100, metadata={"help": "The frequency to save the model relative to the number of training data batches"})
    ppo_epochs: Optional[int] = field(default=100, metadata={"help": "The number of PPO epochs"})
    train_proportion: Optional[float] = field(default=0.7, metadata={"help": "The proportion of the data to use for training"})
    number_shared_layers_for_ref_model: Optional[int] = field(default=0, metadata={"help": "The number of shared layers between the model and the reference model"})
    create_ref_model: Optional[bool] = field(default=False, metadata={"help": "Whether to create a reference model for PPO RLHF"})

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Load configuration from a YAML file and create an instance of ScriptArguments.
        Missing fields in the YAML will be completed with the class defaults.
        """
        # Load YAML file
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Initialize the class using the loaded YAML data
        return cls(**yaml_data)



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
    

