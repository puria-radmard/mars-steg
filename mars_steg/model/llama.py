from mars_steg.model.base_model import BaseModel
from typing import Dict


class Llama_31_8B_Instruct(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
    
class Llama_32_1B_Instruct(Llama_31_8B_Instruct):
    pass
