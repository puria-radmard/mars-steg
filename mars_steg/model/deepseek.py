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
    
    def full_generate(self, conversations_list: Conversation):

        for conversation in conversations_list:
            conversation.append({'role': 'assistant', 'content': "<think> Okay, so I need to figure out if the student's final answer is correct based on the provided correct solution so Im going to give the direct Answer in decimal values from 0.0 to 1.0 and Im NOT going to break it step by step and reply directly. Ending thinking "})

        transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
        inputs = self.tokenizer(transformed_conversations, return_tensors="pt").to("cuda")
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
