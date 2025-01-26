import torch as t
import numpy as np
import random
from torch import nn
import warnings
from typing import Optional, Dict
from torch.optim import Adam

from typing import List, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer
from mars_steg.config import ModelConfig, OptimizerConfig, PromptConfig
from mars_steg.utils.exceptions import LLMTranscriptExtractionError
from mars_steg.utils.score import CoTGapSummary
from mars_steg.model import BaseModel, GPT2Model, Llama_31_8B_Instruct, Llama_32_1B_Instruct, DeepSeek_R1_1pt5B

from torch.utils.data import DataLoader

from mars_steg.utils.prompt_data import BatchPromptData

from mars_steg import task
from mars_steg import language
from mars_steg.task import TorchDatasetTask
from mars_steg.language import LanguageAspect

from trl import create_reference_model



def get_prompt_config(config_dir):
    prompt_config = PromptConfig()
    prompt_config.from_file(config_dir)
    return prompt_config


def get_dataloaders_and_ref_model(
    dataset_class_name: str,
    penalisation_class_name: str,
    dataset_class_kwargs: Optional[Dict],
    prompt_config: Optional[Dict],
    penalisation_class_kwargs: Optional[Dict],
    train_proportion: float,
    validation_proportion: float,
    batch_size: int,
    training_model: Optional[BaseModel] = None,        # TODO: turn this into BaseModel!
    tokenizer: Optional[AutoTokenizer] = None,
    generation_kwargs: Optional[Dict] = None,
    *_,
    override_create_ref_model: bool = False,
    number_shared_layers_for_ref_model: Optional[int] = None
):
    """
    Including ref model creation here because it is needed for the task,
    as well as potentially for the training!

    Create the reference model if `override_create_ref_model` is True
    or if the neural overseer or task requires it
    """

    dataset_class_kwargs = dataset_class_kwargs or dict()
    penalisation_class_kwargs = penalisation_class_kwargs or dict()
    print(dataset_class_name)
    print(dataset_class_kwargs)
    print(prompt_config)
    language_aspect_class: LanguageAspect = getattr(language, penalisation_class_name)(**penalisation_class_kwargs)
    task_class: TorchDatasetTask = getattr(task, dataset_class_name)(language_aspect = language_aspect_class, prompt_config = prompt_config, **dataset_class_kwargs )

    need_reference_model = (
        override_create_ref_model or 
        language_aspect_class.uses_local_neural_overseer or
        task_class.uses_local_neural_assessor
    )
    if need_reference_model:
        reference_model = create_reference_model(
            training_model.model, num_shared_layers=number_shared_layers_for_ref_model,
            pattern = 'pretrained_model.base_model.model.model.layers.{layer}'  # XXX: this works for peft LoRA, not tried for other things
        )
    else:
        reference_model = None

    if language_aspect_class.uses_local_neural_overseer:
        neural_overseer: BaseModel = training_model.duplicate(reference_model)   # TODO: takes in new (frozen) model, and adds own config/kwargs etc. to new instance
        language_aspect_class.recruit_neural_overseer(reference_model, tokenizer, generation_kwargs)    # TODO: change this to accept BaseModel 

    if task_class.uses_local_neural_assessor:
        neural_assesor: BaseModel = training_model.duplicate(reference_model)   # TODO: takes in new (frozen) model, and adds own config/kwargs etc. to new instance
        task_class.recruit_neural_assessor(reference_model, tokenizer, generation_kwargs)   # TODO: change this to accept BaseModel 

    train_dataset, val_dataset, test_dataset = task_class.test_train_split(
        train_proportion=train_proportion,
        validation_proportion=validation_proportion
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=task_class.prompt_data_collate_fn)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_class.prompt_data_collate_fn)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=task_class.prompt_data_collate_fn)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, reference_model



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
    print(f'GETTING: {model_name}')
    if model_name == "gpt2":
        return GPT2Model(model_config=model_config, tokenizer=tokenizer)
    elif model_name == 'meta-llama/Llama-3.1-8B-Instruct':
        return Llama_31_8B_Instruct(model_config=model_config, tokenizer=tokenizer)
    elif model_name == 'meta-llama/Llama-3.2-1B-Instruct':
        return Llama_32_1B_Instruct(model_config=model_config, tokenizer=tokenizer)
    elif model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B':
        return DeepSeek_R1_1pt5B(model_config=model_config, tokenizer=tokenizer)
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
    system_prompt: str, user_prompts: List[str], model_class: BaseModel
):
    batch_messages = [
        [
            model_class.get_message(
                "system", system_prompt
            ),  # {"role": "system", "content": system_prompt},
            model_class.get_message(
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
    

def extract_cots(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]):
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


def extract_neural_overseer_answers(batch_prompt_datas: BatchPromptData, output_delimitation_tokens: Dict[str, str]):
    """
    Take in full batch of prompts, and spit out batch of prompts for which the overseer model was able to give
        a well formatted answer. If output length is less than input length, this means some responses could
        not be parsed
    """
    extracted_prompt_datas = []
    for prompt_data in tqdm(batch_prompt_datas):
        try:
            prompt_data.extracted_overseer_answer = parse_model_output(prompt_data.overseer_transcript, output_delimitation_tokens, cot=False)
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
            prompt_data.extracted_assessor_answer = parse_model_output(prompt_data.assessor_transcript, output_delimitation_tokens, cot=False)
            extracted_prompt_datas.append(prompt_data)
        except LLMTranscriptExtractionError as e:
            warnings.warn(f"Failed to parse assessor answer from: \n\n{prompt_data.assessor_transcript}\n\n Skipping example")
    return BatchPromptData(extracted_prompt_datas)


def get_rewards_and_training_datas(
        batch_prompt_datas: BatchPromptData, 
        query_tensors: List[t.TensorType], 
        transcript_responses: List[t.TensorType], 
        train_dataset: TorchDatasetTask, 
        ) -> Tuple[List]:
    """
    batch_prompt_datas come in imbued with answers extracted from the neural overseer and assesor already
    Any that failed should have been pruned from this argument anyway, so no need to double check here! 

    However: Just because the extraction worked does not mean that the score assignment will
    We do another round of pruning here if the extracted answers in either NOS or NA are not in the right format
    """
    
    composite_reward_list, task_score_list, language_score_list = [], [], []
    train_query_tensors = []
    train_transcript_responses = []

    if batch_assessor_scores is None:
        batch_assessor_scores = [None]*len(batch_prompt_datas)

    for i, prompt_data in tqdm(enumerate(batch_prompt_datas)):
        
        composite_reward = 0.0
        task_score = 0.0
        language_score = 0.0

        try:
            composite_reward, task_score, language_score = train_dataset.reward_from_transcript(
                prompt_data,
            )
        except LLMTranscriptExtractionError:
            continue

        composite_reward_list.append(t.tensor(composite_reward))
        task_score_list.append(task_score)
        language_score_list.append(language_score)
        train_query_tensors.append(query_tensors[i])
        train_transcript_responses.append(transcript_responses[i])

    return composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses
    
            
            



def evaluate_cot_gap_summary(
        ppo_trainer, 
        tokenizer, 
        dataloader: DataLoader,
        system_prompt: str,
        model_class: BaseModel, 
        device, 
        generation_kwargs: Dict,
        output_delimitation_tokens: Dict[str, str],
    ) -> CoTGapSummary:
    
    # Total examples provided:
    sum_examples = 0

    # Examples answered in the correct format:
    sum_extracted_with_cot = 0
    sum_extracted_without_cot = 0

    # Of those answered of the correct format...
    sum_reward_with_cot = 0
    sum_reward_without_cot = 0
    
    with t.no_grad():

        for batch_prompt_datas in tqdm(dataloader):

            batch_messages_without_cot = batchize_conversation(
                system_prompt, batch_prompt_datas.no_cot_prompts, model_class
            )

            batch_messages_with_cot = batchize_conversation(
                system_prompt, batch_prompt_datas.cot_prompts, model_class
            )
            batch_length = len(batch_messages_with_cot)
            batch_messages = batch_messages_with_cot + batch_messages_without_cot
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
                truncation=True,
            )

            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Translate it into ppo.generate format which List[torch.tensors]
            query_tensors = []
            for input in inputs["input_ids"]:
                query_tensors.append(input)

            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            transcript_responses = ppo_trainer.generate(
                query_tensors, **generation_kwargs,
            )
            decoded_responses = [
                tokenizer.decode(r.squeeze()) for r in transcript_responses
            ]

            batch_prompt_datas.cot_transcripts = decoded_responses[:batch_length]
            batch_prompt_datas.no_cot_transcripts = decoded_responses[batch_length:]

            # TODO: this
            raise NotImplementedError('Need to incorporate neural assessor for rewards here!')

            for prompt_data in batch_prompt_datas:
                sum_examples += 1

                try:
                    # We don't actually need the CoT prompt
                    _, prompt_data.extracted_final_answer_with_cot = parse_model_output(prompt_data.cot_transcript, output_delimitation_tokens, cot = True)
                    sum_extracted_with_cot += 1
                    sum_reward_with_cot += dataloader.dataset.get_task_score(prompt_data, with_cot=True)
                except LLMTranscriptExtractionError:
                    warnings.warn(f"Failed to parse: \n\n{prompt_data.cot_transcript}\n\n (with CoT) Skipping example")

                try:
                    _, prompt_data.extracted_final_answer_without_cot = parse_model_output(prompt_data.no_cot_transcript, output_delimitation_tokens, cot = False)
                    sum_extracted_without_cot += 1
                    sum_reward_without_cot += dataloader.dataset.get_task_score(prompt_data, with_cot=False)
                except LLMTranscriptExtractionError:
                    warnings.warn(f"Failed to parse: \n\n{prompt_data.no_cot_transcript}\n\n (without CoT) Skipping example")

            cot_gap = CoTGapSummary(
                sum_examples,
                sum_extracted_with_cot,
                sum_extracted_without_cot,
                sum_reward_with_cot,
                sum_reward_without_cot,
            )

            print(cot_gap.summary)

    # cot_gap = (sum_reward_with_cot - sum_reward_without_cot)
    # cot_ratio = sum_reward_with_cot / (sum_reward_without_cot + t.finfo(t.float32).eps)

    return CoTGapSummary(
        sum_examples,
        sum_extracted_with_cot,
        sum_extracted_without_cot,
        sum_reward_with_cot,
        sum_reward_without_cot,
    )

    
