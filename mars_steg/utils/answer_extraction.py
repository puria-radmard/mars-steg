import re

from tqdm import tqdm
from typing import Dict, Optional
from mars_steg.utils.prompt_data import BatchPromptData
from mars_steg.utils.exceptions import LLMTranscriptExtractionError

import warnings


END_TOKENS = [
    "<｜end▁of▁sentence｜>",
    "<|end_of_text|>",
    "</OUTPUT>",
    "<|eot_id|>",
]


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
        if cot:
            if scratchpad_start == -1:
                raise LLMTranscriptExtractionError(
                    "CoT section could not be parsed from the prompt."
                )
            chain_of_thought = extract_section(scratchpad_start,start_scratchpad_token,scratchpad_end,model_output)
        else:
            chain_of_thought = None
    
    elif not (start_scratchpad_token or end_scratchpad_token) and start_output_token:

        output_start = model_output.rfind(start_output_token)
        if output_start == -1:
            raise LLMTranscriptExtractionError(
                "Final answer section could not be parsed from the prompt."
            )
        if not end_output_token:
            output = extract_section(output_start,start_output_token,len(model_output),model_output)
        else:
            output_end = model_output.find(end_output_token)
            output = extract_section(output_start,start_output_token,output_end,model_output)
        
        if cot:
            chain_of_thought = extract_section(0,start_output_token, output_start,model_output)
        else:
            chain_of_thought = None
        import pdb; pdb.set_trace(header="check this!")

    elif start_scratchpad_token and end_scratchpad_token and not (start_output_token or end_output_token):

        raise NotImplementedError('This needs to be debugged!')

        scratchpad_start = model_output.find(start_scratchpad_token)
        scratchpad_end = model_output.find(end_scratchpad_token)

        # Extract the output
        if output_start == -1 or output_end == -1 or output_start >= output_end:
            raise LLMTranscriptExtractionError(
                "OUTPUT section could not be parsed from the prompt."
            )
        
        if scratchpad_end == -1:
            output = extract_section(output_start,0,len(model_output),model_output)
        else:
            output = extract_section(output_start,scratchpad_end,len(model_output),model_output)

        # Extract the scratchpad reasoning
        chain_of_thought = None

        if cot:
            if scratchpad_start == -1:
                raise LLMTranscriptExtractionError(
                    "CoT section could not be parsed from the prompt."
                )
            chain_of_thought = extract_section(scratchpad_start,start_scratchpad_token,scratchpad_end,model_output)
        
    for et in END_TOKENS:
        output = output.replace(et, "")
        
    return chain_of_thought, output

    


def extract_neural_overseer_answers(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]):
    """
    Take in full batch of prompts, and spit out batch of prompts for which the overseer model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed
    """
    extracted_prompt_datas = []
    for prompt_data in tqdm(batch_prompt_datas):
        try:
            _, prompt_data.extracted_overseer_answer = parse_model_output(prompt_data.overseer_transcript, output_delimitation_tokens, cot=False)
            extracted_prompt_datas.append(prompt_data)
        except LLMTranscriptExtractionError as e:
            warnings.warn(f"Failed to parse overseer answer from: \n\n{prompt_data.overseer_transcript}\n\n Skipping example")
    return BatchPromptData(extracted_prompt_datas)


def extract_neural_assessor_answers(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]):
    """
    Take in full batch of prompts, and spit out batch of prompts for which the assessor model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed
    """
    extracted_prompt_datas = []
    for prompt_data in tqdm(batch_prompt_datas):
        try:
            import pdb; pdb.set_trace(header="check this!")
            _, prompt_data.extracted_assessor_answer = parse_model_output(prompt_data.assessor_transcript, output_delimitation_tokens, cot=False)
            extracted_prompt_datas.append(prompt_data)
        except LLMTranscriptExtractionError as e:
            warnings.warn(f"Failed to parse assessor answer from: \n\n{prompt_data.assessor_transcript}\n\n Skipping example")
    return BatchPromptData(extracted_prompt_datas)

def extract_cots(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]) -> BatchPromptData:
    """
    Take in full batch of prompts, and spit out batch of prompts for which the model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed
    """
    extracted_prompt_datas = []
    for prompt_data in tqdm(batch_prompt_datas):
        try:
            (
                prompt_data.extracted_cot,
                prompt_data.extracted_final_answer_with_cot,
            ) = parse_model_output(prompt_data.cot_transcript, output_delimitation_tokens, cot=True)
            extracted_prompt_datas.append(prompt_data)
        except LLMTranscriptExtractionError as e:
            warnings.warn(f"Failed to parse: \n\n{prompt_data.cot_transcript}\n\n Skipping example")
    return BatchPromptData(extracted_prompt_datas)



def extract_boolean_overseer_or_assessor_answer(
    response_text: str
) -> float:
    """
    Gets a boolean value
    TODO: generalise to a float [0,1]
    """
    response_lower = response_text.lower()
    rfind_true = response_lower.rfind('true')
    rfind_false = response_lower.rfind('false')
    if rfind_true == rfind_false:
        raise LLMTranscriptExtractionError(f'Neural overseer did not return a TRUE or FALSE token at the end of its response\n: {response_text}')
    return 1.0 if bool(rfind_true > rfind_false) else 0.0


float_pattern = re.compile('(?<![0-9])(0\\.[0-9]+|1\\.0+(?![0-9]))')


def extract_floatingpoint_overseer_or_assessor_answer(
    response_text: str
) -> float:
    """
    Gets a float value
    TODO: check if ret > 1.0
    TODO: improvement of regex
    TODO: solve system prompt being replicated in response
    """
    import pdb; pdb.set_trace(header="extract_floatingpoint_overseer_or_assessor_answer")
    response_lower = response_text.lower().rstrip('}')
    match = re.findall(float_pattern, response_lower)[-1]
    return float(match)
   
