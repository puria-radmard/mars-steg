from mars_steg.model.base_model import BaseModel
from typing import Dict


class Llama_31_8B_Instruct(BaseModel):
    """
    A subclass of BaseModel for handling the specific implementation of the Llama 31.8B Instruct model.
    This class implements the method for formatting messages used by the Llama 31.8B Instruct model.

    Methods
    -------
    get_message(user_string: str, system_prompt: str) -> Dict
        Formats and returns a message dictionary with the specified user role and system prompt.
    """

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        """
        Generates a message dictionary containing the role and content for a conversation. 
        The role is determined by the user_string and the content is the system prompt.

        Parameters
        ----------
        user_string : str
            The role of the user in the conversation (e.g., 'user', 'assistant').
        system_prompt : str
            The content of the message, typically the system's prompt to the user.

        Returns
        -------
        Dict
            A dictionary containing the 'role' and 'content' for the message.
        """
        return {"role": user_string, "content": f"{system_prompt}"}

    
class Llama_32_1B_Instruct(Llama_31_8B_Instruct):
    """
    A subclass of Llama_31_8B_Instruct for handling the specific implementation of the Llama 32.1B Instruct model.
    Inherits behavior from Llama_31_8B_Instruct without modification.

    Methods
    -------
    Inherits `get_message` from Llama_31_8B_Instruct without changes.
    """
    pass