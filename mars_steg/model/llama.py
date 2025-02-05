from mars_steg.model.base_model import BaseModel
from typing import Dict, List

Conversation = List[List[Dict[str, str]]]

class Llama_31_8B_Instruct(BaseModel):
    
    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
        
    def full_generate(self, conversations_list: Conversation):

        inputs = self.tokenizer.apply_chat_template(
            conversations_list,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
            padding=True,
            truncation=True,
        )
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        generation_kwargs = self.generation_config.to_dict()
        exclude_keys = ['return_prompt', 'generate_ref_response', 'batch_size']
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if k not in exclude_keys}
        generation_kwargs['eos_token_id'] = generation_kwargs['pad_token_id']
        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
        )

        prompt_length = inputs['input_ids'].shape[1]
        decoded_responses = [
            self.tokenizer.decode(output[prompt_length:]) 
            for output in outputs
        ]

        return decoded_responses

class Llama_32_1B_Instruct(Llama_31_8B_Instruct):
    pass