import torch as t
import numpy as np
import random
from typing import Optional

def get_device() -> str:
    if t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    #if t.cuda.is_available() and t.cuda.device_count() > 1:
        #raise ValueError("Please specify which GPU to use with CUDA_VISIBLE_DEVICES")

    return device

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False


class LLMTranscriptExtractionError(Exception):
    pass


cot_start_token = None


        
def extract_section(output_start: int, start_section_token: str, output_end: int ,model_output: str):
    '''
    Extract a section from the output of the model based on start and end positions.

    Args:
        output_start (int): The starting index of the section in the model output.
        start_section_token (str): The token indicating the beginning of the section.
        output_end (int): The ending index of the section in the model output. 
                          If -1, the section extends to the end of the output.
        model_output (str): The complete output string from the model.

    Returns:
        str: The extracted section of the output, stripped of surrounding whitespace.
    '''
    
    output_start_idx = output_start + len(start_section_token)
    output_end_idx = output_end if output_end != -1 else len(model_output)
    output = model_output[output_start_idx:output_end_idx].strip()
    return output


# Make a config so we can define the tokens to separate cot and answers on LLM responses
def parse_model_output(model_output: str, cot: bool = False, output_delimitation_tokens: dict = None) -> tuple[Optional[str], str]:
    """
    Parses a given model output into a tuple of (scratchpad_reasoning, output).
    
    Args:
        model_output (str): The input string containing SCRATCHPAD_REASONING and OUTPUT sections
        cot (bool): If True, raises an error if SCRATCHPAD_REASONING cannot be parsed
        output_delimitation_tokens (dict): If defined, defines the tokens of the expected sections
        
    Returns:
        tuple[str | None, str]: A tuple containing (scratchpad_reasoning, output)
            scratchpad_reasoning will be None if cot=False
        
    Raises:
        ValueError: If cot=True and SCRATCHPAD_REASONING cannot be parsed,
                  or if OUTPUT section cannot be parsed
    """

    if not output_delimitation_tokens:
        raise ValueError("Invalid token configuration provided in output_delimitation_tokens. Please provide the tokens.")
    
    
    start_output_token = output_delimitation_tokens.get("start_output_token")
    end_output_token = output_delimitation_tokens.get("end_output_token")
    start_scratchpad_token = output_delimitation_tokens.get("start_scratchpad_token")
    end_scratchpad_token = output_delimitation_tokens.get("end_scratchpad_token")
    

    if start_output_token and end_output_token and start_scratchpad_token and end_scratchpad_token:
        
        output_start = model_output.find(start_output_token)
        output_end = model_output.find(end_output_token)
        scratchpad_start = model_output.find(start_scratchpad_token)
        scratchpad_end = model_output.find(end_scratchpad_token)

        # Extract the output
        if output_start == -1 or output_end == -1 or output_start >= output_end:
            raise LLMTranscriptExtractionError(
                "OUTPUT section could not be parsed from the prompt."
            )

        output = extract_section(output_start,start_output_token,output_end,model_output)

        # Extract the scratchpad reasoning
        chain_of_thought = None

        if cot:
            if scratchpad_start == -1:
                raise LLMTranscriptExtractionError(
                    "SCRATCHPAD_REASONING section could not be parsed from the prompt."
                )

            chain_of_thought = extract_section(scratchpad_start,start_scratchpad_token,scratchpad_end,model_output)
        
        output = output.replace("<|end_of_text|>", "")
        output = output.replace("</OUTPUT>", "")
        
        return chain_of_thought, output
    
    elif not start_scratchpad_token and not end_scratchpad_token  and start_output_token:

        output_start = model_output.find(start_output_token)
        if output_start == -1:
            raise LLMTranscriptExtractionError(
                "OUTPUT section could not be parsed from the prompt."
            )
        if not end_output_token:
            output = extract_section(output_start,start_output_token,len(model_output),model_output)
        else:
            output_end = model_output.find(end_output_token)
            output = extract_section(output_start,start_output_token,output_end,model_output)
        chain_of_thought = None
        if cot:
            chain_of_thought = extract_section(0,start_output_token, output_start,model_output)
        
        output = output.replace("<|end_of_text|>", "")
        output = output.replace("</OUTPUT>", "")
        
        return chain_of_thought, output
    
            
            



   


        