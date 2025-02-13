from __future__ import annotations
import warnings

from mars_steg.model.base_model import BaseModel
from typing import Dict, List

Conversation = List[List[Dict[str, str]]]

class DeepSeek_R1_1pt5B(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        if "system" in user_string:
            user_string = "assistant"

        return {"role": user_string, "content": f"{system_prompt}"}
    
    @staticmethod
    def transform_conversation(conversations_list: Conversation):
        conversation = ""
        for message in conversations_list:
            role = message["role"]
            content = message["content"]
            if role == "user":
                conversation += f"<｜User｜> {content}\n"
            elif role == "assistant":
                conversation += f"<｜Assistant｜> {content}\n"  
        return conversation
    
    def full_generate(self, 
                      conversations_list: Conversation, 
                      is_neural_assessor: bool = False, 
                      is_neural_overseer: bool = False, 
                      neural_assessor_thinking_helper: str = None,
                      neural_overseer_thinking_helper: str = None):

        if is_neural_overseer or is_neural_assessor:
            if neural_overseer_thinking_helper is None and is_neural_overseer:
                warnings.warn("No neural_overseer_thinking_helper provide, the model might produce useless tokens that could lead it to hallucinate")
            elif neural_assessor_thinking_helper is None and is_neural_assessor:
                warnings.warn("No neural_assessor_thinking_helper provide, the model might produce useless tokens that could lead it to hallucinate")
            else:
                for conversation in conversations_list:
                    conversation.append(
                        {
                        'role': 'assistant', 
                        'content': neural_overseer_thinking_helper if is_neural_overseer else neural_assessor_thinking_helper

                        }
                    )

        transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
        print(transformed_conversations)
        inputs = self.tokenizer(transformed_conversations, return_tensors="pt").to(self.device)
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}

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
