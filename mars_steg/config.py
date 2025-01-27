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
class PromptConfig:

    # TODO: logic for constructing with task, examples, cot, and answer format

    # Task dependent
    # TODO: Absorb system_prompt in mars_steg/task/tasks/math_dataset_task.py
        # EXCLUDING the answer format instructions
    task_description_dir: Optional[str] = None
    task_description: Optional[str] = None

    # Instructions on CoT usage
    with_cot_dir: Optional[str] = None
    with_cot: Optional[str] = None
    no_cot_dir: Optional[str] = None
    no_cot: Optional[str] = None

    # Task dependent
    # TODO: Absorb examples in mars_steg/task/tasks/math_dataset_task.py
    task_examples_dir: Optional[str] = None
    task_examples: Optional[str] = None

    # Instructions on answer
    # TODO: Absorb the answer format instructions in system_prompt in mars_steg/task/tasks/math_dataset_task.py
    answer_format_dir: Optional[str] = None
    answer_format: Optional[str] = None

    # XXX: General template, needs other things slotted in
    # TODO: mars_steg/language/language_aspects/neural_overseer.py --> DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_TEMPLATE
    # TODO: slotting in logic
    neural_overseer_instruction_dir: Optional[str] = None
    neural_overseer_instruction: Optional[str] = None

    # TODO: slot into general format in mars_steg/language/language_aspects/neural_overseer.py
        # REPLACE: "Make sure this boolean is the last token you output."
    neural_overseer_answer_format_dir: Optional[str] = None
    neural_overseer_answer_format: Optional[str] = None

    # XXX: TODO: open up space for this!
    neural_assessor_instruction_dir: Optional[str] = None
    neural_assessor_instruction: Optional[str] = None
    neural_assessor_answer_format_dir: Optional[str] = None
    neural_assessor_answer_format: Optional[str] = None

    # Delimination
    start_scratchpad_token_dir: Optional[str] = None
    start_scratchpad_token: Optional[str] = None
    end_scratchpad_token_dir: Optional[str] = None
    end_scratchpad_token: Optional[str] = None
    start_output_token_dir: Optional[str] = None
    start_output_token: Optional[str] = None
    end_output_token_dir: Optional[str] = None
    end_output_token: Optional[str] = None


    @classmethod
    def from_file(cls, file_path: str):
        """
        XXX: *super* hacky
        Initialise all as Nones
        Replace all of those with config entries
        Delete everything that remains None --> attrbute error if used later
        """
        config_dict = load_yaml(file_path)
        instance = cls()
        
        for name, dir_ in config_dict.items():
            attr_name = name.replace("_dir", "")
            assert getattr(instance, name) is None
            assert getattr(instance, attr_name) is None
            setattr(instance, name, dir_)
            with open(dir_, 'r') as f:
                setattr(instance, attr_name, f.read().strip())
        
        for name, attr in instance.__dict__.items():
            if attr is None:
                delattr(instance, name)
        
        return instance

    #def __getattribute__(self, __name: str):

        #if self.__getattr__(__name) == None:
            #raise ValueError(f'PromptConfig not given a {__name} field')
 
    # def __setattribute__(self, name : str, value: str):
    #     # set attribute to value
    #     if not isinstance(value,str):
    #         raise ValueError(f'The value of {name} should be a string')
        




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

class PromptConfigLoader:

    # TODO: loading
    # TODO: task prompt construction
    # TODO: NOS prompt construction
    # TODO: NA prompt construction
        # Requires communication with other subgroup
    # TODO: provide to relevant classes and raise exceptions where needed

    @staticmethod
    def load_config(yaml_file: str) -> PromptConfig:
        config_dict = load_yaml(yaml_file)
        prompt_dir = config_dict["PromptDir"]
        for name, dir_ in prompt_dir:
            attr_name = name.replace("_dir", "")
        return prompt_config



class ConfigLoader:

    @staticmethod
    def load_config(yaml_file: str) -> Tuple[ModelConfig, OptimizerConfig, TrainConfig, GenerationConfig]:
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

if __name__ == "__main__":
    CONFIG_DIR = "mars_steg/configs/config.yaml"
    prompt_config = PromptConfig()
    prompt_config.from_file(CONFIG_DIR)
    print(prompt_config.task_description_dir)
    print(prompt_config.task_examples)
    print(prompt_config.task_examples_dir)
