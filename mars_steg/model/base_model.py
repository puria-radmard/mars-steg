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

from transformers import BitsAndBytesConfig

import warnings


class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_config: ModelConfig, tokenizer: AutoTokenizer):
        self.model_name = model_config.model_name
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.lora = model_config.lora
        self.model_save_path = model_config.model_save_path
        self.model = self.load_model()

    def load_model(self):
        precission_mode = self.model_config.load_precision_mode
        if "4bits" in precission_mode:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        elif "8bits" in precission_mode:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
        )
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        warnings.warn('Reintroduce: model.resize_token_embeddings(len(self.tokenizer)) ?')
        if self.lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, **self.model_config.lora_params
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        else:
            warnings.warn('Reintroduce LoRA!')
        return model

    @abstractmethod
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        pass
