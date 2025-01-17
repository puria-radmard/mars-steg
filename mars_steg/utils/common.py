import torch as t
import numpy as np
import random
from torch import nn
from typing import Optional, Dict
from torch.optim import Adam

from typing import List

from transformers import AutoTokenizer
from mars_steg.config import ModelConfig, OptimizerConfig
from mars_steg.utils.exceptions import LLMTranscriptExtractionError
from mars_steg.model import BaseModel, GPT2Model, Llama_31_8B_Instruct

from torch.utils.data import DataLoader

from mars_steg import task
from mars_steg import language
from mars_steg.task import TorchDatasetTask
from mars_steg.language import LanguageAspect


def get_dataloaders(
    dataset_class_name: str,
    penalisation_class_name: str,
    dataset_class_kwargs: Optional[Dict],
    penalisation_class_kwargs: Optional[Dict],
    train_proportion: float,
    batch_size: int
):

    dataset_class_kwargs = dataset_class_kwargs or dict()
    penalisation_class_kwargs = penalisation_class_kwargs or dict()

    language_aspect_class: LanguageAspect = getattr(language, penalisation_class_name)(**penalisation_class_kwargs)
    task_class: TorchDatasetTask = getattr(task, dataset_class_name)(language_aspect = language_aspect_class, **dataset_class_kwargs)

    train_dataset, test_dataset = task_class.test_train_split(
        train_proportion=train_proportion
    )
    train_loader, test_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_class.prompt_data_collate_fn,
    ), DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_class.prompt_data_collate_fn,
    )

    return train_dataset, test_dataset, train_loader, test_loader




def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_optimizer(optimizer_config: OptimizerConfig, model: nn.Module) -> t.optim.Optimizer:
    if optimizer_config.optimizer_name == "adam":
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_config.learning_rate,
        )
        return optimizer


def get_model(model_config: ModelConfig, tokenizer: AutoTokenizer) -> BaseModel:
    model_name = model_config.model_name
    if model_name == "gpt2":
        return GPT2Model(model_config=model_config, tokenizer=tokenizer)
    if "meta-llama/Llama" in  model_name:
        return Llama_31_8B_Instruct(model_config=model_config, tokenizer=tokenizer)
    
    else:
        raise ValueError(model_name)






def get_device() -> str:
    if t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    # if t.cuda.is_available() and t.cuda.device_count() > 1:
    # raise ValueError("Please specify which GPU to use with CUDA_VISIBLE_DEVICES")

    return device


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False


def batchize_conversation(
    system_prompt: str, user_prompts: List[str], model: BaseModel
):
    batch_messages = [
        [
            model.get_message(
                "system", system_prompt
            ),  # {"role": "system", "content": system_prompt},
            model.get_message(
                "user", user_prompt
            ),  # {"role": "user", "content": user_prompt}
        ]
        for user_prompt in user_prompts
    ]
    return batch_messages

        
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
def parse_model_output(model_output: str, output_delimitation_tokens: Dict[str, str], cot: bool = False) -> tuple[Optional[str], str]:
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
                    "CoT section could not be parsed from the prompt."
                )

            chain_of_thought = extract_section(scratchpad_start,start_scratchpad_token,scratchpad_end,model_output)
        
        output = output.replace("<|end_of_text|>", "")
        output = output.replace("</OUTPUT>", "")
        output = output.replace("<|eot_id|>", "")
        
        return chain_of_thought, output
    
    elif not start_scratchpad_token and not end_scratchpad_token  and start_output_token:

        output_start = model_output.find(start_output_token)
        if output_start == -1:
            raise LLMTranscriptExtractionError(
                "Final answer section could not be parsed from the prompt."
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
        output = output.replace("<|eot_id|>", "")        
        
        return chain_of_thought, output
    
            
            



   


        
