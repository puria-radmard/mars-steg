from __future__ import annotations
import torch
from mars_steg.model.base_model import BaseModel
from typing import Dict, List
import warnings

Conversation = List[List[Dict[str, str]]]

class Llama_31_8B_Instruct(BaseModel):
    
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
    
    @staticmethod
    def transform_conversation(conversations_list, prompt_thinking_helper: str = None)-> Conversation:
        warnings.warn("No option for thinking helper for LLama models implemented. STOP THE PROGRAM IF YOU EXPECTED TO HAVE IT")
        return conversations_list
    
    def tokenize(self, conversations_list: Conversation) -> List[Dict[str, torch.TensorType]]:
        return self.tokenizer.apply_chat_template(
            conversations_list,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            truncation=True,
        ).to(self.device)

class Llama_32_1B_Instruct(Llama_31_8B_Instruct):
    pass