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
    
    Parameters
    ----------
    output_start : int
        The starting index of the section in the model output.
    start_section_token : str
        The token indicating the beginning of the section.
    output_end : int
        The ending index of the section in the model output. If -1, the section extends to the end of the output.
    model_output : str
        The complete output string from the model.
    
    Parameters
    ----------
    output_start : int
        The starting index of the section in the model output.
    start_section_token : str
        The token indicating the beginning of the section.
    output_end : int
        The ending index of the section in the model output. If -1, the section extends to the end of the output.
    model_output : str
        The complete output string from the model.

    Returns
    -------
    str
        The extracted section of the output, stripped of surrounding whitespace.
    Returns
    -------
    str
        The extracted section of the output, stripped of surrounding whitespace.
    '''
    
    output_start_idx = output_start + len(start_section_token)
    output_end_idx = output_end if output_end != -1 else len(model_output)
    output = model_output[output_start_idx:output_end_idx].strip()
    return output



# Make a config so we can define the tokens to separate cot and answers on LLM responses
def parse_model_output(model_output: str, output_delimitation_tokens: Dict[str, str], cot: bool = False) -> tuple[Optional[str], str]:
    
    
    """
    Parses a given model output into a tuple of (scratchpad_reasoning, output).

    Parameters
    ----------
    model_output : str
        The input string containing SCRATCHPAD_REASONING and OUTPUT sections
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.
    cot : bool
        If True, raises an error if SCRATCHPAD_REASONING cannot be parsed

    Returns
    -------
    tuple[str | None, str]
        A tuple containing (scratchpad_reasoning, output) scratchpad_reasoning will be None if cot=False
    Parameters
    ----------
    model_output : str
        The input string containing SCRATCHPAD_REASONING and OUTPUT sections
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.
    cot : bool
        If True, raises an error if SCRATCHPAD_REASONING cannot be parsed

    Returns
    -------
    tuple[str | None, str]
        A tuple containing (scratchpad_reasoning, output) scratchpad_reasoning will be None if cot=False

    Raises
    ------
    ValueError
        If cot=True and SCRATCHPAD_REASONING cannot be parsed or if OUTPUT section cannot be parsed
    Raises
    ------
    ValueError
        If cot=True and SCRATCHPAD_REASONING cannot be parsed or if OUTPUT section cannot be parsed
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

    elif start_scratchpad_token and end_scratchpad_token and not (start_output_token or end_output_token):

        scratchpad_start = model_output.find(start_scratchpad_token)
        scratchpad_end = model_output.find(end_scratchpad_token)
        output = extract_section(scratchpad_end,end_scratchpad_token, len(model_output), model_output)

        if scratchpad_start == -1:
            warnings.warn(f"{start_scratchpad_token} not found in the decoded answer, might be in thinking helper. If not, stop the training process RIGHT NOW")
            scratchpad_start = 0
            start_scratchpad_token = ""
            

        # Extract the scratchpad reasoning
        chain_of_thought = None

        if cot:
            if scratchpad_start == -1:
                raise LLMTranscriptExtractionError(
                    "CoT section could not be parsed from the prompt."
                )
            chain_of_thought = extract_section(scratchpad_start,start_scratchpad_token,scratchpad_end,model_output)

    else:
        raise NotImplementedError("""
                The output delimitation tokens defined are not falling into one of the above:
                Cases that are implemented : 
                    - start_output_token and end_output_token and start_scratchpad_token and end_scratchpad_token
                    - not (start_scratchpad_token or end_scratchpad_token) and start_output_token
                    - start_scratchpad_token and end_scratchpad_token and not (start_output_token or end_output_token)""")
        
    for et in END_TOKENS:
        output = output.replace(et, "")
        
    return chain_of_thought, output

    


def extract_neural_overseer_answers(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]):
    
    
    """
    Take in full batch of prompts, and spit out batch of prompts for which the OVERSEER model was able to give
    Take in full batch of prompts, and spit out batch of prompts for which the OVERSEER model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed

    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the overseer_transcript attribute set (this will be parsed, to set extracted_overseer_answer)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_overseer_answer attribute set

    """


    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the overseer_transcript attribute set (this will be parsed, to set extracted_overseer_answer)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_overseer_answer attribute set

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
    Take in full batch of prompts, and spit out batch of prompts for which the ASSESSOR model was able to give
    Take in full batch of prompts, and spit out batch of prompts for which the ASSESSOR model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed

    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the assessor_transcript attribute set (this will be parsed, to set extracted_assessor_answer)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_assessor_answer attribute set

    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the assessor_transcript attribute set (this will be parsed, to set extracted_assessor_answer)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_assessor_answer attribute set
    """
    extracted_prompt_datas = []
    for prompt_data in tqdm(batch_prompt_datas):
        try:
            _, prompt_data.extracted_assessor_answer = parse_model_output(prompt_data.assessor_transcript, output_delimitation_tokens, cot=False)
            extracted_prompt_datas.append(prompt_data)
        except LLMTranscriptExtractionError as e:
            warnings.warn(f"Failed to parse assessor answer from: \n\n{prompt_data.assessor_transcript}\n\n Skipping example")
    return BatchPromptData(extracted_prompt_datas)



def extract_cots(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]) -> BatchPromptData:
    """
    Take in full batch of prompts, and spit out batch of prompts for which the MODEL BEING TRAINED was able to give
    Take in full batch of prompts, and spit out batch of prompts for which the MODEL BEING TRAINED was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed

    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the cot_transcript attribute set (this will be parsed, to set extracted_cot and extracted_final_answer_with_cot)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_cot and extracted_final_answer_with_cot attributes set


    Parameters
    ----------
    batch_prompt_data : str
        An instance of BatchPromptData. Must have the cot_transcript attribute set (this will be parsed, to set extracted_cot and extracted_final_answer_with_cot)
    output_delimitation_tokens : Dict[str, str]
        A dict containing the tokens which will be used to delineate parts of the model's responses. 
        For the main model being trained, which emits a scratchpad and final output, the keys should be:
        - "start_output_token"
        - "end_output_token"
        - "start_scratchpad_token"
        - "end_scratchpad_token"
        And the values should be an appropriately chosen token, e.g <SCRATCHPAD_REASONING>, </SCRATCHPAD_REASONING> etc.

    Returns
    -------
    BatchPromptData
        A new version of the input BatchPromptData, now with the extracted_cot and extracted_final_answer_with_cot attributes set

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
    For cases where the answer given by a model is given as the character strings 'TRUE' or 'FALSE'. Not case-sensitive. 
    Expects that this token will be very close to end of text and represent final answer. 
    Finds the last instance of either of these strings in the response text and returns 1.0 for TRUE or 0.0 for false. 
    Throws an LLMTranscriptExtractionError if neither string is present. 

    Parameters
    ----------
    response_text : str
        The response text from a model, which should contain substrings 'TRUE' or 'FALSE'. 

    Returns
    -------
    float
        1.0 for TRUE or 0.0 for FALSE

    Raises
    ------
    LLMTranscriptExtractionError
        If neither 'TRUE' nor 'FALSE' is found in string

    For cases where the answer given by a model is given as the character strings 'TRUE' or 'FALSE'. Not case-sensitive. 
    Expects that this token will be very close to end of text and represent final answer. 
    Finds the last instance of either of these strings in the response text and returns 1.0 for TRUE or 0.0 for false. 
    Throws an LLMTranscriptExtractionError if neither string is present. 

    Parameters
    ----------
    response_text : str
        The response text from a model, which should contain substrings 'TRUE' or 'FALSE'. 

    Returns
    -------
    float
        1.0 for TRUE or 0.0 for FALSE

    Raises
    ------
    LLMTranscriptExtractionError
        If neither 'TRUE' nor 'FALSE' is found in string

    """
    response_lower = response_text.lower()
    rfind_true = response_lower.rfind('true')
    rfind_false = response_lower.rfind('false')
    if rfind_true == rfind_false:
        raise LLMTranscriptExtractionError(f'Neural overseer did not return a TRUE or FALSE token at the end of its response: \n\n{response_text}')
        raise LLMTranscriptExtractionError(f'Neural overseer did not return a TRUE or FALSE token at the end of its response: \n\n{response_text}')
    return 1.0 if bool(rfind_true > rfind_false) else 0.0


float_pattern = re.compile('(?<![0-9])(0\\.[0-9]*|1\\.0*(?![0-9])|1(?![0-9]|\\.)|0(?![0-9]))')

def extract_floatingpoint_overseer_or_assessor_answer(
    response_text: str
) -> float:
    """
    For cases where the answer given by a model is given as a float in range [0,1]. 
    Expects that this score is the LAST example of such a float in the model's answer. 
    Finds the last instance of a float, specifically in this range, and returns it, cast as a float. 
    Throws an LLMTranscriptExtractionError if neither string is present. 

    For cases where the answer given by a model is given as a float in range [0,1]. 
    Expects that this score is the LAST example of such a float in the model's answer. 
    Finds the last instance of a float, specifically in this range, and returns it, cast as a float. 
    Throws an LLMTranscriptExtractionError if neither string is present. 

    TODO: check if ret > 1.0
    TODO: solve system prompt being replicated in response

    Parameters
    ----------
    response_text : str
        The response text from a model, which should contain a float in range [0-1].

    Returns
    -------
    float
        1.0 for TRUE or 0.0 for FALSE

    Raises
    ------
    LLMTranscriptExtractionError
        If no example of a float in range [0-1] is found in the string

    Notes
    -----
    Relies on the float pattern above. 
    This captures:
    1) Any number with a '0', then a '.', and then zero or more digits, matching as many as possible. 
    2) Any number with a '1', then a '.', and then zero or more '0's, matching as many as possible, but refusing if any digits follow which are not 0 (hence score > 1). 
    3) A single solitary '0', refusing if it is followed by any further digits.
    4) A single solitary '1', refusing if it is followed by any further digits, or a '.' (which would be covered by case 2) 

    All cases 1-4 are also refused if they are PRECEDED by any other digits (i.e score > 10 when should be < 1)


    Parameters
    ----------
    response_text : str
        The response text from a model, which should contain a float in range [0-1].

    Returns
    -------
    float
        1.0 for TRUE or 0.0 for FALSE

    Raises
    ------
    LLMTranscriptExtractionError
        If no example of a float in range [0-1] is found in the string

    Notes
    -----
    Relies on the float pattern above. 
    This captures:
    1) Any number with a '0', then a '.', and then zero or more digits, matching as many as possible. 
    2) Any number with a '1', then a '.', and then zero or more '0's, matching as many as possible, but refusing if any digits follow which are not 0 (hence score > 1). 
    3) A single solitary '0', refusing if it is followed by any further digits.
    4) A single solitary '1', refusing if it is followed by any further digits, or a '.' (which would be covered by case 2) 

    All cases 1-4 are also refused if they are PRECEDED by any other digits (i.e score > 10 when should be < 1)

    """
    response_lower = response_text.lower().rstrip('}')
    matches = re.findall(float_pattern, response_lower)
    if not matches:
        raise LLMTranscriptExtractionError(f'Neural overseer did not find an example of a float in range [0-1] in the response text below: \n\n{response_text}')
    return float(matches[-1])

   
