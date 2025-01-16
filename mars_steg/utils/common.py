import torch as t
import numpy as np
import random
from typing import Optional
from torch.optim import Adam

from transformers import AutoTokenizer
from mars_steg.config import ModelConfig, OptimizerConfig
from mars_steg.model import BaseModel, GPT2Model

def get_tokenizer(model_name: str):
    tokenizer =  AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_optimizer(optimizer_config: OptimizerConfig, model) -> t.optim.Optimizer:
    if optimizer_config.optimizer_name == "adam":
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer_config.learning_rate,)
        return optimizer
    
def get_model(model_config: ModelConfig, tokenizer: AutoTokenizer) -> BaseModel:
    model_name = model_config.model_name
    if model_name == "gpt2":
        return GPT2Model(model_config= model_config, tokenizer= tokenizer)
    else:
        raise ValueError

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

def test_parser():
    # Test 1: Basic case with both tags (should succeed)
    test1 = """
    <SCRATCHPAD_REASONING>
    This is some reasoning
    </SCRATCHPAD_REASONING>
    <OUTPUT>
    This is the output
    </OUTPUT>
    """
    reasoning, out = parse_model_output(test1, cot=True)
    assert reasoning == "This is some reasoning", f"Expected 'This is some reasoning', got '{reasoning}'"
    assert out == "This is the output", f"Expected 'This is the output', got '{out}'"

    # Test 2: Missing scratchpad but cot=False (should succeed)
    test2 = """
    <OUTPUT>
    Just output, no reasoning
    </OUTPUT>
    """
    reasoning, out = parse_model_output(test2, cot=False)
    assert reasoning is None, f"Expected None, got '{reasoning}'"
    assert out == "Just output, no reasoning", f"Expected 'Just output, no reasoning', got '{out}'"

    # Test 3: Missing scratchpad with cot=True (should fail)
    test3 = """
    <OUTPUT>
    This should raise an error when cot=True
    </OUTPUT>
    """
    try:
        parse_model_output(test3, cot=True)
        assert False, "Expected LLMTranscriptExtractionError for missing SCRATCHPAD_REASONING with cot=True"
    except LLMTranscriptExtractionError:
        pass  # This is expected

    # Test 4: Missing end tags (should fail)
    test4 = """
    <SCRATCHPAD_REASONING>
    Incomplete reasoning
    <OUTPUT>
    Incomplete output
    """
    try:
        parse_model_output(test4, cot=True)
        assert False, "Expected LLMTranscriptExtractionError for missing end tags"
    except LLMTranscriptExtractionError:
        pass  # This is expected

    # Test 5: Empty tags (should succeed but with empty strings)
    test5 = """
    <SCRATCHPAD_REASONING>
    </SCRATCHPAD_REASONING>
    <OUTPUT>
    </OUTPUT>
    """
    reasoning, out = parse_model_output(test5, cot=True)
    assert reasoning == "", f"Expected empty string, got '{reasoning}'"
    assert out == "", f"Expected empty string, got '{out}'"

    print("All tests passed!")

if __name__ == "__main__":
    test_parser()
