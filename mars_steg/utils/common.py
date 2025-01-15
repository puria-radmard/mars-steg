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

    output_start = model_output.find("<OUTPUT>")
    output_end = model_output.find("</OUTPUT>")
    scratchpad_start = model_output.find("<SCRATCHPAD_REASONING>")
    scratchpad_end = model_output.find("</SCRATCHPAD_REASONING>")

    # Extract the output
    if output_start == -1 or output_end == -1 or output_start >= output_end:
        raise LLMTranscriptExtractionError("OUTPUT section could not be parsed from the prompt.")
    output_start_idx = output_start + len("<OUTPUT>")
    output_end_idx = output_end if output_end != -1 else len(model_output)
    output = model_output[output_start_idx:output_end_idx].strip()


    # Extract the scratchpad reasoning
    chain_of_thought = None
    if cot:
        if scratchpad_start == -1:
            raise LLMTranscriptExtractionError("SCRATCHPAD_REASONING section could not be parsed from the prompt.")
        scratchpad_start_idx = scratchpad_start + len("<SCRATCHPAD_REASONING>")
        scratchpad_end_idx = scratchpad_end if scratchpad_end != -1 else output_start
        chain_of_thought = model_output[scratchpad_start_idx:scratchpad_end_idx].strip()

    return chain_of_thought, output
