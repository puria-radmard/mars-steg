from dataclasses import dataclass
import yaml
from typing import Tuple
from dataclasses import asdict
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    model_save_path: str
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
    number_shared_layers_for_ref_model: int
    create_ref_model: bool


class ConfigLoader:
    @staticmethod
    def load_yaml(yaml_file: str) -> dict:
        """
        Load a YAML file into a dictionary.
        """
        file_path = f"{yaml_file}"
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_config(yaml_file: str) -> Tuple[ModelConfig, OptimizerConfig, TrainConfig]:
        """
        Load YAML data and map it to configuration dataclasses.
        """
        config_dict = ConfigLoader.load_yaml(yaml_file)

        # Create instances of the dataclasses
        model_config = ModelConfig(**config_dict["Model"])
        optimizer_config = OptimizerConfig(**config_dict["Optimizer"])
        train_config = TrainConfig(**config_dict["Train"])

        return model_config, optimizer_config, train_config

    @staticmethod
    def save_config(
        yaml_file: str,
        model_config: ModelConfig,
        optimizer_config: OptimizerConfig,
        train_config: TrainConfig,
    ):
        """
        Save the dataclass configurations back to a YAML file.
        """
        config_dict = {
            "Model": asdict(model_config),
            "Optimizer": asdict(optimizer_config),
            "Train": asdict(train_config),
        }
        with open(yaml_file, "w") as file:
            yaml.dump(config_dict, file)
