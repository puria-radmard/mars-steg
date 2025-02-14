from __future__ import annotations
import warnings

import torch

from mars_steg.model.base_model import BaseModel
from typing import Dict, List, Optional

Conversation = List[List[Dict[str, str]]]

class DeepSeek_R1_1pt5B(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        if "system" in user_string:
            user_string = "assistant"

        return {"role": user_string, "content": f"{system_prompt}"}
    
    def tokenize(self, conversations_list: List[List[str]]) -> List[Dict[str, torch.TensorType]]:
            return self.tokenizer(
                conversations_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
    
    @staticmethod
    def transform_conversation(conversations_list: Conversation, prompt_thinking_helper: Optional[str] = None):
        conversation = ""
        for message in conversations_list:
            role = message["role"]
            content = message["content"]
            if role == "user":
                conversation += f"<｜User｜> {content}\n"
            elif role == "assistant":
                conversation += f"<｜Assistant｜> {content}\n"  
        if prompt_thinking_helper is not None:
            conversation += f"<｜Assistant｜> {prompt_thinking_helper}\n"
        return conversation
    