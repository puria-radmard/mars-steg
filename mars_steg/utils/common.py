import torch as t
import numpy as np
import random
from typing import Optional
from dataclasses import dataclass


# Type checking [0,1] scores
def check_score(score_name, score_value):
    if not isinstance(score_value, (float, bool, int)):
        raise TypeError(f'.get_{score_name}() must bool, or float in range [0,1], or ints 0 or 1. Returned type: {type(score_value)}')
    score_value = float(score_value)
    if score_value < 0 or score_value > 1: 
        raise ValueError(f'.get_{score_name}() returned value {score_value} after conversion to float, which is outside range [0,1]')
    return score_value


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

def parse_model_output(model_output: str, cot: bool = False) -> tuple[Optional[str], str]:
    """
    Parses a given model output into a tuple of (scratchpad_reasoning, output).
    
    Args:
        model_output (str): The input string containing SCRATCHPAD_REASONING and OUTPUT sections
        cot (bool): If True, raises an error if SCRATCHPAD_REASONING cannot be parsed
        
    Returns:
        tuple[str | None, str]: A tuple containing (scratchpad_reasoning, output)
            scratchpad_reasoning will be None if cot=False
        
    Raises:
        ValueError: If cot=True and SCRATCHPAD_REASONING cannot be parsed,
                  or if OUTPUT section cannot be parsed
    """
    # Extract the scratchpad reasoning
    chain_of_thought = None
    if cot:
        scratchpad_start = model_output.find("<SCRATCHPAD_REASONING>")
        scratchpad_end = model_output.find("</SCRATCHPAD_REASONING>")
        if scratchpad_start == -1 or scratchpad_end == -1 or scratchpad_start >= scratchpad_end:
            raise ValueError("SCRATCHPAD_REASONING section could not be parsed from the prompt.")
        scratchpad_start_idx = scratchpad_start + len("<SCRATCHPAD_REASONING>")
        chain_of_thought = model_output[scratchpad_start_idx:scratchpad_end].strip()

    # Extract the output
    output_start = model_output.find("<OUTPUT>")
    output_end = model_output.find("</OUTPUT>")
    if output_start == -1 or output_end == -1 or output_start >= output_end:
        raise ValueError("OUTPUT section could not be parsed from the prompt.")
    output_start_idx = output_start + len("<OUTPUT>")
    output = model_output[output_start_idx:output_end].strip()

    return chain_of_thought, output
