from mars_steg.model.base_model import BaseModel
from typing import Dict, List

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
    
    def transform_conversation(self, conversations_list: Conversation):
        conversation = ""
        for message in conversations_list:
            role = message["role"]
            content = message["content"]
            if role == "user":
                conversation += f"<｜User｜> {content}\n"
            elif role == "assistant":
                conversation += f"<｜Assistant｜> {content}\n"  
        return conversation
    
    def full_generate(self, conversations_list: Conversation) -> List[str]:
        """
        Generates responses for the provided conversations. The assistant's message is 
        appended to each conversation before generating responses.

        Parameters
        ----------
        conversations_list : Conversation
            A list of conversations for which responses are to be generated.

        Returns
        -------
        List[str]
            A list of decoded responses from the model.
        """
        
        # Append a message from the assistant regarding the expected output format
        for conversation in conversations_list:
            conversation.append({'role': 'assistant', 'content': "<think> Okay, so I need to figure out if the student's final answer is correct based on the provided correct solution so Im going to give the direct Answer in decimal values from 0.0 to 1.0 and Im NOT going to break it step by step and reply directly. Ending thinking "})

        # Transform the conversations into the desired format for input
        transformed_conversations = [self.transform_conversation(conversation) for conversation in conversations_list]
        
        # Tokenize the transformed conversations and move to the appropriate device
        inputs = self.tokenizer(transformed_conversations, return_tensors="pt").to("cuda")
        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        
        # Create a new dictionary for generation parameters
        generation_kwargs = self.generation_config.to_generation_dict()
        {k: v for k, v in self.generation_config.to_dict().items() if k not in ['return_prompt', 'generate_ref_response', 'batch_size']}

        # Run the model to generate responses
        outputs = self.model.generate(**inputs, **generation_kwargs)

        # Decode the responses from the model output
        decoded_responses = [
            # TODO: self.tokenizer.decode(r.squeeze()[prompt_length:]) for prompt_length, r in zip(prompt_lengths, outputs)
            self.tokenizer.decode(r.squeeze()) for r in outputs
        ]
        
        return decoded_responses