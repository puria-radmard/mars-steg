from mars_steg.config import ModelConfig
from typing import Dict, List
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
    def __init__(self, model_config: ModelConfig, tokenizer: AutoTokenizer, output_delimitation_tokens: Dict[str, str], model=None):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.tokenizer = tokenizer
        self.lora = model_config.lora
        self.precission_mode = model_config.load_precision_mode
        self.model_save_path = model_config.model_save_path
        self.output_delimitation_tokens = output_delimitation_tokens
        if model is not None:
            self.model = model
        else:
            self.model = self.load_model()

    def load_model(self):
        self.log_loading()
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
    
    def log_loading(self):
        print("------------------")
        print(f"Model loaded in {self.precission_mode} mode")
        print("------------------")
    
    def duplicate(self, reference_model):
        # Create a new instance of the same class
        new_instance = self.__class__(
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            output_delimitation_tokens = self.output_delimitation_tokens,
            model=reference_model if reference_model is not None else None
        )

        new_instance.model.eval()
        for param in new_instance.model.parameters():
            param.requires_grad = False
            
        return new_instance

    @abstractmethod
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        pass

    def batchize_conversation(
        self, system_prompt: str, user_prompts: List[List[Dict[str, str]]]
    ):
        batch_messages = [
            [
                self.get_message(
                    "system", system_prompt
                ),  # {"role": "system", "content": system_prompt},
                self.get_message(
                    "user", user_prompt
                ),  # {"role": "user", "content": user_prompt}
            ]
            for user_prompt in user_prompts
        ]
        return batch_messages
