from mars_steg.config import ModelConfig
from typing import Dict
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from abc import ABCMeta, abstractmethod
import torch
from trl import AutoModelForCausalLMWithValueHead
import warnings
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
    warnings.warn('Reintroduce: BitsAndBytesConfig. Import hasn\'t been successful.')




class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_config: ModelConfig, tokenizer: AutoTokenizer):
        self.model_name = model_config.model_name
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.lora = model_config.lora
        self.precission_mode = model_config.load_precision_mode
        self.model_save_path = model_config.model_save_path
        self.model = self.load_model()

    def load_model(self):
        print("------------------")
        print(f"Model loaded in {self.precission_mode} mode")
        print("------------------")
        quantization_config = {
            "4bits": BitsAndBytesConfig(load_in_4bit=True),
            "8bits": BitsAndBytesConfig(load_in_8bit=True)
        }.get(self.precission_mode, None)

        warnings.warn('Reintroduce: model.resize_token_embeddings(len(self.tokenizer)) ?')
        if self.lora:
            lora_config = LoraConfig(
                task_type="CAUSAL_LM", **self.model_config.lora_params
            )
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                self.model_name,
                peft_config=lora_config,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                quantization_config=quantization_config,
            )
        return model

    @abstractmethod
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        pass
