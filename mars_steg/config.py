from __future__ import annotations

from dataclasses import dataclass
import yaml
from typing import Callable, Tuple
from dataclasses import asdict
from typing import Optional, Dict


def load_yaml(yaml_file: str) -> dict:
    """
    Load a YAML file into a dictionary.
    """
    file_path = f"{yaml_file}"
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    model_save_path: str
    load_precision_mode: str
    lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str

    @property
    def lora_params(self):
        return {
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.lora_bias,
        }


@dataclass
class OptimizerConfig:
    optimizer_name: str
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    max_grad_norm: float



@dataclass
class ExperimentArgs:
    dataset_class_name: str
    penalisation_class_name: str
    dataset_class_kwargs: Dict
    penalisation_class_kwargs: Dict

    @classmethod
    def from_yaml(cls, yaml_file: str) -> ExperimentArgs:
        config_dict = load_yaml(yaml_file)
        return cls(**config_dict)



@dataclass
class TrainConfig:
    log_with: Optional[str]  # e.g., "wandb" or None
    mini_batch_size: int
    batch_size: int
    gradient_accumulation_steps: int
    seed: int
    num_train_epochs: int
    num_eval_episodes: int
    save_frequency: int
    ppo_epochs: int
    train_proportion: float
    validation_proportion: float
    number_shared_layers_for_ref_model: int
    create_ref_model: bool
    evaluate_cot_gap: bool
    number_of_evaluations: int  

@dataclass
class GenerationConfig:
    min_length: int
    num_beams: int
    do_sample: bool
    max_new_tokens: int
    return_prompt: bool
    generate_ref_response: bool
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    length_sampler: Optional[Callable] = None
    batch_size: Optional[int] = None
    pad_token_id: Optional[int] = None
    

    def __post_init__(self):
        if not self.do_sample:
            assert self.temperature == self.top_k == self.top_p == None,\
                "Cannot specify GenerationConfig.temperature if GenerationConfig.do_sample is specified"
    
    def to_dict(self):
        return asdict(self)

    # @classmethod
    # def from_yaml(cls, yaml_file):
    #     with open(yaml_file, "r") as file:
    #         data = yaml.safe_load(file)
    #     return cls(**data)

    # def to_yaml(self, yaml_file):
    #     with open(yaml_file, "w") as file:
    #         yaml.dump(asdict(self), file)

class ConfigLoader:

    @staticmethod
    def load_config(yaml_file: str) -> Tuple[ModelConfig, OptimizerConfig, TrainConfig]:
        """
        Load YAML data and map it to configuration dataclasses.
        """
        config_dict = load_yaml(yaml_file)

        # Create instances of the dataclasses
        model_config = ModelConfig(**config_dict["Model"])
        optimizer_config = OptimizerConfig(**config_dict["Optimizer"])
        train_config = TrainConfig(**config_dict["Train"])
        generation_config = GenerationConfig(**config_dict["Generation"])

        return model_config, optimizer_config, train_config, generation_config
    

    @staticmethod
    def save_config(
        yaml_file: str,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        train_config: TrainConfig,
        generation_config: GenerationConfig
    ):
        """
        Save the dataclass configurations back to a YAML file.
        """
        config_dict = {
            "Model": asdict(model_config),
            "Optimizer": asdict(optimizer_config),
            "Train": asdict(train_config),
            "Generation": asdict(generation_config),
        }
        with open(yaml_file, "w") as file:
            yaml.dump(config_dict, file)
