from __future__ import annotations
import warnings

import torch

from mars_steg.model.base_model import BaseModel
from typing import Dict, List, Optional

Conversation = List[List[Dict[str, str]]]

class DeepSeek_R1_1pt5B(BaseModel):
    
    """
    A subclass of BaseModel that implements specific methods for handling 
    conversation data and generating responses using a model.

    Methods
    -------
    get_message(user_string: str, system_prompt: str) -> Dict
        Formats a message based on the user string and system prompt.
    transform_conversation(conversations_list: Conversation) -> str
        Transforms a list of conversations into a single formatted string.
    full_generate(conversations_list: Conversation) -> List[str]
        Generates responses for a list of conversations using the model.
    """
    
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        """
        Generates a message dictionary based on the user string and system prompt. 
        If the user string contains 'system', it is replaced with 'assistant'.

        Parameters
        ----------
        user_string : str
            The role of the user in the conversation.
        system_prompt : str
            The system prompt to include in the message content.

        Returns
        -------
        Dict
            A dictionary containing the role and content of the message.
        """
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
        """
        Transforms a list of conversation messages into a formatted string. 
        The conversation will be formatted by roles (user or assistant).

        Parameters
        ----------
        conversations_list : Conversation
            A list of conversation messages where each message is a dictionary with 'role' and 'content'.

        Returns
        -------
        str
            A formatted string representing the entire conversation.
        """
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
    
