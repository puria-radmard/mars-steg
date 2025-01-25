from mars_steg.model.base_model import BaseModel
from typing import Dict


class DeepSeek_R1_1pt5B(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
