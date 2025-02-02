"""
Module Name: config.py
------------------------


Description:
------------
This module contains the configuration dataclasses for the model, optimizer, training, and generation configurations.


Data Classes:
- ModelConfig: Contains the model configuration parameters.
- OptimizerConfig: Contains the optimizer configuration parameters.
- TrainConfig: Contains the training configuration parameters.
- GenerationConfig: Contains the generation configuration parameters.
- PromptConfig: Contains the prompt configuration parameters.
- ExperimentArgs: Contains the experiment configuration parameters.

Classes:
- ConfigLoader: Contains the methods to load and save the configuration dataclasses to a YAML file.
- PromptConfigLoader: Contains the methods to load the prompt configuration dataclass from a YAML


"""

from __future__ import annotations

from dataclasses import dataclass
import yaml
from typing import Callable, Tuple
from dataclasses import asdict
from typing import Optional, Dict, Any


def load_yaml(yaml_file: str) -> dict:
    """        
        Parameters
        ----------
        yaml_file : str
            The path to the YAML file.

        Returns
        -------
        dict
            The contents of the YAML file as a dictionary.
    """

    file_path = f"{yaml_file}"
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass for model configuration parameters.

    Attributes
    ----------
    model_name : str
        The name of the model.
    model_save_path : str
        The path to save the model.
    load_precision_mode : str
        The precision mode for loading the model.
    lora : bool
        Whether to use LoRA.
    lora_r : int
        The LoRA parameter r.
    lora_alpha : int
        The LoRA parameter alpha.
    lora_dropout : float
        The LoRA dropout rate.
    lora_bias : str
        The LoRA bias.

    Properties
    ----------
    lora_params : dict
        The LoRA parameters as a dictionary.

    See Also
    --------

    #TODO : Add references for LoRA
    
    """
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
    """
    Dataclass for optimizer configuration parameters.

    Attributes
    ----------
    optimizer_name : str
        The name of the optimizer.
    learning_rate : float
        The learning rate.
    weight_decay : float
        The weight decay.
    adam_epsilon : float
        The Adam epsilon.
    warmup_steps : int
        The warmup steps.
    max_grad_norm : float
        The maximum gradient norm.

    """

    optimizer_name: str
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_steps: int
    max_grad_norm: float

@dataclass
class PromptConfig:
    """
    Dataclass for prompt configuration parameters.

    Attributes
    ----------
    task_description_dir : Optional[str]
        The path to the task description directory.
    task_description : Optional[str]
        The task description.
    with_cot_instruction_dir : Optional[str]
        The path to the CoT instruction directory.
    with_cot_instruction : Optional[str]
        The CoT instruction with CoT.
    no_cot_instruction_dir : Optional[str]
        The path to the "No CoT" instruction directory.
    no_cot_instruction : Optional[str]
        The "No CoT" instruction.
    task_examples_dir : Optional[str]
        The path to the task examples directory to put in the prompt of an LLM to give him example formats to follow.
    task_examples : Optional[str]
        The task examples.

    answer_format_instruction_dir : Optional[str]
        The path to the answer format instruction directory.
    answer_format_instruction : Optional[str]
        The answer format instruction.

    neural_overseer_user_prompt_template_dir : Optional[str]
        The path to the neural overseer user prompt template directory.
    neural_overseer_user_prompt_template : Optional[str]
        The neural overseer user prompt template.
    neural_overseer_system_prompt_dir : Optional[str]
        The path to the neural overseer system prompt directory.
    neural_overseer_system_prompt : Optional[str]
        The neural overseer system prompt.

    neural_assessor_user_prompt_template_dir : Optional[str]
        The path to the neural assessor user prompt template directory.
    neural_assessor_user_prompt_template : Optional[str]
        The neural assessor user prompt template.
    neural_assessor_system_prompt_dir : Optional[str]
        The path to the neural assessor system prompt directory.
    neural_assessor_system_prompt : Optional[str]
        The neural assessor system prompt.
    
    start_scratchpad_token_dir : Optional[str]
        The path to the start scratchpad token directory.
    start_scratchpad_token : Optional[str]
        The start scratchpad token.
    end_scratchpad_token_dir : Optional[str]
        The path to the end scratchpad token directory.
    end_scratchpad_token : Optional[str]
        The end scratchpad token.
    start_output_token_dir : Optional[str]
        The path to the start output token directory.
    start_output_token : Optional[str]
        The start output token.
    end_output_token_dir : Optional[str]
        The path to the end output token directory.
    end_output_token : Optional[str]
        The end output token.

    Methods
    -------
    from_file(cls, file_path: str) -> PromptConfig
        Load the prompt configuration dataclass from a YAML
    __getattribute__(self, __name: str) -> Any
        Get the attribute value of the dataclass. Override the default __getattribute__ method.

    Notes
    -----

    Example folder can be found in the `mars_steg/prompts`, containing texts files for each attribute.

    """

    # TODO: logic for constructing with task, examples, cot, and answer format

    # Task dependent
    # TODO: Absorb system_prompt in mars_steg/task/tasks/math_dataset_task.py
        # EXCLUDING the answer format instructions
    task_description_dir: Optional[str] = None
    task_description: Optional[str] = None

    # Instructions on CoT usage
    with_cot_instruction_dir: Optional[str] = None
    with_cot_instruction: Optional[str] = None
    no_cot_instruction_dir: Optional[str] = None
    no_cot_instruction: Optional[str] = None

    # Task dependent
    # TODO: Absorb examples in mars_steg/task/tasks/math_dataset_task.py
    task_examples_dir: Optional[str] = None
    task_examples: Optional[str] = None

    # Instructions on answer
    # TODO: Absorb the answer format instructions in system_prompt in mars_steg/task/tasks/math_dataset_task.py
    answer_format_instruction_dir: Optional[str] = None
    answer_format_instruction: Optional[str] = None
    prompt_thinking_helper_dir: Optional[str] = None
    prompt_thinking_helper: Optional[str] = None


    neural_overseer_user_prompt_template_dir: Optional[str] = None
    neural_overseer_user_prompt_template: Optional[str] = None
    neural_overseer_system_prompt_dir: Optional[str] = None
    neural_overseer_system_prompt: Optional[str] = None
    neural_overseer_thinking_helper_dir : Optional[str] = None
    neural_overseer_thinking_helper : Optional[str] = None

    neural_assessor_user_prompt_template_dir: Optional[str] = None
    neural_assessor_user_prompt_template: Optional[str] = None
    neural_assessor_system_prompt_dir: Optional[str] = None
    neural_assessor_system_prompt: Optional[str] = None
    neural_assessor_thinking_helper_dir : Optional[str] = None
    neural_assessor_thinking_helper : Optional[str] = None

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
            assert super(PromptConfig, instance).__getattribute__(name) is None
            assert super(PromptConfig, instance).__getattribute__(attr_name) is None
            setattr(instance, name, dir_)
            with open(dir_, 'r') as f:
                setattr(instance, attr_name, f.read().strip())

        return instance

    def __getattribute__(self, __name: str):

        ret = super().__getattribute__(__name)
        #TODO Check if it's right to have added the last bool condition -> returns None if we haven't set up a directory to the token, otherwise it raises an error
        if (ret == None) and ('token' not in __name) and (super().__getattribute__(__name + "_dir") is not None):
            raise ValueError(f'PromptConfig not given a {__name} field')

        return ret
 
    # def __setattribute__(self, name : str, value: str):
    #     # set attribute to value
    #     if not isinstance(value,str):
    #         raise ValueError(f'The value of {name} should be a string')
        




@dataclass
class ExperimentArgs:
    """
    Dataclass for experiment configuration parameters.

    Attributes
    ----------
    dataset_class_name : str
        The name of the dataset class.
    penalisation_class_name : str
        The name of the penalisation class.
    dataset_class_kwargs : Dict
        The dataset class keyword arguments.
    penalisation_class_kwargs : Dict
        The penalisation class keyword arguments.

    Methods
    -------
    from_yaml(cls, yaml_file: str) -> ExperimentArgs
        Load the experiment configuration dataclass from a YAML


    """
    dataset_class_name: str
    penalisation_class_name: str
    dataset_class_kwargs: Dict
    penalisation_class_kwargs: Dict

    @classmethod
    def from_yaml(cls, yaml_file: str) -> ExperimentArgs:
        """
        Load the experiment configuration dataclass from a YAML

        Parameters
        ----------
        yaml_file : str
            The path to the YAML file.

        Returns
        -------
        ExperimentArgs
            The experiment configuration dataclass.

        """
        config_dict = load_yaml(yaml_file)
        return cls(**config_dict)


@dataclass
class TrainConfig:
    """
    Dataclass for training configuration parameters.

    Attributes
    ----------
    log_with : Optional[str]
        The logging tool to use. Choices are "wandb" or None.
    mini_batch_size : int
        The mini batch size for PPO trainer.
    batch_size : int
        The batch size.
    gradient_accumulation_steps : int
        The gradient accumulation steps.
    seed : int
        The seed.
    num_train_epochs : int
        The number of training epochs.
    num_eval_episodes : int
        The number of evaluation episodes.
    save_frequency : int
        The save frequency.
    ppo_epochs : int
        The PPO epochs.
    train_proportion : float
        The training proportion.
    validation_proportion : float
        The validation proportion.
    number_shared_layers_for_ref_model : int
        The number of shared layers for the reference model.
    create_ref_model : bool
        Whether to create the reference model.
    evaluate_cot_gap : bool
        Whether to evaluate the CoT gap.
    number_of_evaluations : int
        The number of evaluations.

    Notes
    -----
    See huggingface's `transformers` library for more information on the PPO trainer.

    .. [1] #TODO: Add reference to huggingface's `transformers` library


    """

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
    """
    Dataclass for generation configuration parameters.

    Attributes
    ----------
    min_length : int
        The minimum length when generating for an LLM.
    num_beams : int
        The number of beams when generating for an LLM.
    do_sample : bool
        Whether to sample when generating for an LLM.
    max_new_tokens : int
        The maximum number of new tokens to generate.
    return_prompt : bool
        Whether to return the prompt.
    generate_ref_response : bool
        Whether to generate the reference response.
    temperature : Optional[float]
        The temperature.
    top_k : Optional[int]
        The top k.
    top_p : Optional[int]
        The top p.
    length_sampler : Optional[Callable]
        The length sampler.
    batch_size : Optional[int]
        The batch size.
    pad_token_id : Optional[int]
        The pad token id.
    bos_token_id : Optional[int]
        The bos token id.
    eos_token_id : Optional[int]
        The eos token id.

    Notes
    -----

    See huggingface's `transformers` library for more information on the generation configuration.

    .. [1] #TODO: Add reference to huggingface's `transformers` library
        
    """

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
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: Optional[bool] = None
    
    # return_full_text: bool = False

    def __post_init__(self):
        if not self.do_sample:
            assert self.temperature == self.top_k == self.top_p == None,\
                "Cannot specify GenerationConfig.temperature if GenerationConfig.do_sample is specified"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_generation_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.to_dict().items() if k not in ['return_prompt', 'generate_ref_response','skip_special_tokens', 'batch_size', 'return_full_text']
        }

    def to_training_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.to_dict().items() if k not in []
        }

    # @classmethod
    # def from_yaml(cls, yaml_file):
    #     with open(yaml_file, "r") as file:
    #         data = yaml.safe_load(file)
    #     return cls(**data)

    # def to_yaml(self, yaml_file):
    #     with open(yaml_file, "w") as file:
    #         yaml.dump(asdict(self), file)

class PromptConfigLoader:

    """
    Class for loading the prompt configuration dataclass from a YAML

    Methods
    -------
    load_config(yaml_file: str) -> PromptConfig
        Load the prompt configuration dataclass from a YAML
    """

    # TODO: loading
    # TODO: task prompt construction
    # TODO: NOS prompt construction
    # TODO: NA prompt construction
        # Requires communication with other subgroup
    # TODO: provide to relevant classes and raise exceptions where needed

    @staticmethod
    def load_config(yaml_file: str) -> PromptConfig:
        """
        Load the prompt configuration dataclass from a YAML

        Parameters
        ----------
        yaml_file : str
            The path to the YAML file.

        Returns
        -------
        PromptConfig
            The prompt configuration dataclass.

        Notes
        -----

        CHECK ALL THE TODO and check for attr_name that is not used
        
        """

        config_dict = load_yaml(yaml_file)
        prompt_dir = config_dict["PromptDir"]
        for name, dir_ in prompt_dir:
            attr_name = name.replace("_dir", "")
        return prompt_config



class ConfigLoader:

    """
    Class for loading and saving the configuration dataclasses to a YAML file.

    Methods
    -------
    load_config(yaml_file: str) -> Tuple[ModelConfig, OptimizerConfig, TrainConfig, GenerationConfig]
        Load YAML data and map it to configuration dataclasses.
    save_config(yaml_file: str, model_config: ModelConfig, optimizer_config: OptimizerConfig, train_config: TrainConfig, generation_config: GenerationConfig)
        Save the dataclass configurations back to a YAML file.

    """

    @staticmethod
    def load_config(yaml_file: str) -> Tuple[ModelConfig, OptimizerConfig, TrainConfig, GenerationConfig]:
        """

        Load YAML data and map it to configuration dataclasses.

        Parameters
        ----------
        yaml_file : str
            The path to the YAML file.

        Returns
        -------
        Tuple[ModelConfig, OptimizerConfig, TrainConfig, GenerationConfig]
            The configuration dataclasses.

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

        Parameters
        ----------
        yaml_file : str
            The path to the YAML file.
        model_config : ModelConfig
            The model configuration dataclass.
        optimizer_config : OptimizerConfig
            The optimizer configuration dataclass.
        train_config : TrainConfig
            The training configuration dataclass.
        generation_config : GenerationConfig
            The generation configuration dataclass.

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
