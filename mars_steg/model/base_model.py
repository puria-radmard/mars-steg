from mars_steg.config import ModelConfig
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from abc import ABCMeta, abstractmethod
import torch
from trl import AutoModelForCausalLMWithValueHead
from mars_steg.config import GenerationConfig
import warnings
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
    warnings.warn('Reintroduce: BitsAndBytesConfig. Import hasn\'t been successful.')


Conversation = List[List[Dict[str, str]]]

class BaseModel(metaclass=ABCMeta):
    def __init__(self, model_config: ModelConfig, tokenizer: AutoTokenizer, generation_config: GenerationConfig, output_delimitation_tokens: Dict[str, str], device: str, model=None):
        self.model_config = model_config
        self.model_name = model_config.model_name
        self.tokenizer = tokenizer
        self.lora = model_config.lora
        self.precission_mode = model_config.load_precision_mode
        self.model_save_path = model_config.model_save_path
        self.output_delimitation_tokens = output_delimitation_tokens
        self.generation_config = generation_config
        if model is not None:
            self.model = model
        else:
            self.model = self.load_model()
        self.device = device
        self.model.to(device)

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
    
    def duplicate(self, whitebox_model):
        # Create a new instance of the same class
        new_instance = self.__class__(
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            output_delimitation_tokens = self.output_delimitation_tokens,
            generation_config = self.generation_config,
            model=whitebox_model if whitebox_model is not None else None,
            device = self.device
        )

        new_instance.model.eval()
        for param in new_instance.model.parameters():
            param.requires_grad = False
            
        return new_instance

    @abstractmethod
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        pass

    def batchize_conversation(
        self, user_prompts: Conversation, system_prompt: Optional[str] = None,
    ):
        if system_prompt is None:
            batch_messages = [
                [
                    self.get_message(
                        "user", user_prompt
                    )  # {"role": "user", "content": user_prompt}
                ]
                for user_prompt in user_prompts
            ]

        else:
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

    def full_generate(self, conversations_list: Conversation):

        # import pdb; pdb.set_trace(header = 'Check every possible prompt - make sure it has all the nec. components!')

        inputs = self.tokenizer.apply_chat_template(
            conversations_list,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            truncation=True,
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        # create a new dict without ['return_prompt', 'generate_ref_response', 'batch_size'] keys
        generation_kwargs = self.generation_config.to_generation_dict()
        {k: v for k, v in self.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the reference model
        #TODO add special token to parse better 
        outputs = self.model.generate(**inputs, **generation_kwargs)

        # These always work, so can diretly set
        decoded_responses = [
            # TODO: self.tokenizer.decode(r.squeeze()[prompt_length:]) for prompt_length, r in zip(prompt_lengths, outputs)
            self.tokenizer.decode(r.squeeze()) for r in outputs
        ]

        return decoded_responses