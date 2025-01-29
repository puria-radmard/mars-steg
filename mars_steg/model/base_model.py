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
        self, system_prompt: str, user_prompts: Conversation
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
        
    def transform_conversation(self,conversations_list: Conversation):
        conversation = ""
        for message in conversations_list:
            print(message)
            role = message["role"]
            content = message["content"]
            
            # Adjust the role tokens: <|User|> for user input, <|Assistant|> for assistant responses
            if role == "user":
                conversation += f"<｜User｜> {content}\n"
            elif role == "assistant":
                conversation += f"<｜Assistant｜> {content}\n"  
        return conversation


    def full_generate(self, conversations_list: Conversation):

        # import pdb; pdb.set_trace(header = 'Check every possible prompt - make sure it has all the nec. components!')

        conversations_list[0].append({'role': 'assistant', 'content': "<think> Okay, so I need to figure out if the student's final answer is correct based on the provided correct solution so Im going to give the direct Answer in decimal values from 0.0 to 1.0 and Im not going to break it step by step "})
        conversation = self.transform_conversation(conversations_list[0])
        print(conversation)
        input_ids = self.tokenizer(conversation, return_tensors="pt").to("cuda")
        # Generate the model's response using the custom generation settings
        generation_kwargs = self.generation_config.to_generation_dict()
        {k: v for k, v in self.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the reference model
        outputs = self.model.generate(input_ids["input_ids"], **generation_kwargs)

        #inputs = self.tokenizer.apply_chat_template(
        #    conversations_list,
        #    return_tensors="pt",
        #    add_generation_prompt=True,
        #    return_dict=True,
        #    padding=True,
        #    truncation=True,
        #)
        #outputs = model.generate(input_ids["input_ids"], max_new_tokens=1024, **generation_config)
        #decoded_inputs = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  # Clean output without special tokens
        assistant_response = response.split('<｜Assistant｜>')[-1].strip()
        print(assistant_response)
        exit()
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

        # create a new dict without ['return_prompt', 'generate_ref_response', 'batch_size'] keys
        generation_kwargs = self.generation_config.to_generation_dict()
        {k: v for k, v in self.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the reference model
        outputs = self.model.generate(**inputs, **generation_kwargs)

        # These always work, so can diretly set
        decoded_responses = [
            # TODO: self.tokenizer.decode(r.squeeze()[prompt_length:]) for prompt_length, r in zip(prompt_lengths, outputs)
            self.tokenizer.decode(r.squeeze()) for r in outputs
        ]

        return decoded_responses
