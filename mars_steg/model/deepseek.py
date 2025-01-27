from mars_steg.model.base_model import BaseModel
from typing import Dict

# deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
class Deepseek_R1_15b_Distill_Qwen(BaseModel):

    def get_message(self, user_string: str, system_prompt: str) -> Dict:
        return {"role": user_string, "content": f"{system_prompt}"}
    
class Deepseek_R1_7b_Distill_Qwen(Deepseek_R1_15b_Distill_Qwen):
    pass

class Deepseek_R1_8b_Distill_Llama(Deepseek_R1_15b_Distill_Qwen):
    pass

