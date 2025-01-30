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
from mars_steg.config import ModelConfig, OptimizerConfig, PromptConfig, GenerationConfig
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




def get_dataloaders_and_ref_model(
    dataset_class_name: str,
    penalisation_class_name: str,
    dataset_class_kwargs: Optional[Dict],
    prompt_config: PromptConfig,
    penalisation_class_kwargs: Optional[Dict],
    train_proportion: float,
    validation_proportion: float,
    batch_size: int,
    training_model: Optional[BaseModel] = None,  
    *_,
    override_create_ref_model: bool = False,
    number_shared_layers_for_ref_model: Optional[int] = None,
    neural_overseer_model: Optional[BaseModel] = None,
    neural_assessor_model: Optional[BaseModel] = None,
):
    """

    A getter for the Task subclasses to be used here; handles splitting into train, val, test. 
    Also returns torch dataloaders for all three. 

    Note - Including ref model creation here because it is needed for the task,
    as well as potentially for the training!

    Create the reference model if `override_create_ref_model` is True
    or if the neural overseer or task requires it

    Parameters
    ----------
    dataset_class_name : str
        The name of the Task subclass used here (defined name, NOT name attribute)
    penalisation_class_name : str
        The name of the LanguageAspect subclass used here
    dataset_class_kwargs : Optional[Dict]
        Kwargs to configure the dataset class being used; these should NOT include (i) language_aspect, or (ii) prompt_config, which will be set by other args.  
    prompt_config : PromptConfig
        PromptConfig to configure the prompt generation in the task subclass. May also be used to configure prompts in LanguageAspect, e.g if this is a NeuralOverseer. 
    penalisation_class_kwargs : Optional[Dict]
        Kwargs to configure the dataset class being used; these should NOT include prompt_config, which will be set automatically. 
    train_proportion : float
        Proportion of dataset to use for train (float in range [0-1])
    validation_proportion : float
        Proportion of dataset to use for validation (float in range [0-1])
    batch_size : int
        Batch size for torch dataloaders
    training_model : Optional[BaseModel], optional
        By default None. Need to pass this if if `override_create_ref_model` is True or if the neural overseer or task requires it
    override_create_ref_model : bool, optional
        By default False. If True, forces creation of a reference model. 
    number_shared_layers_for_ref_model : Optional[int], optional
        Required if reference model created, by default None
    neural_overseer_model : Optional[BaseModel], optional
        If model for neural overseer has already been created, pass here.
    neural_assessor_model : Optional[BaseModel], optional
        If model for neural assessor has already been created, pass here.


    Returns
    -------
    TorchDatasetTask
        train_dataset 
    TorchDatasetTask
        val_dataset
    TorchDatasetTask
        test_dataset
    torch.utils.data.DataLoader
        train_loader
    torch.utils.data.DataLoader
        val_loader
    torch.utils.data.DataLoader
        test_loader
    torch.utils.data.DataLoader
        reference_model
    """

    dataset_class_kwargs = dataset_class_kwargs or dict()
    penalisation_class_kwargs = penalisation_class_kwargs or dict()
    language_aspect_class: LanguageAspect = getattr(language, penalisation_class_name)(**penalisation_class_kwargs, prompt_config=prompt_config)
    task_class: TorchDatasetTask = getattr(task, dataset_class_name)(language_aspect = language_aspect_class, prompt_config = prompt_config, **dataset_class_kwargs)

    need_reference_model = (
        override_create_ref_model or 
        (language_aspect_class.uses_local_neural_overseer and neural_overseer_model is None) or
        (task_class.uses_local_neural_assessor and neural_assessor_model is None)
    )
    if need_reference_model:
        reference_model = create_reference_model(
            training_model.model, num_shared_layers=number_shared_layers_for_ref_model,
            pattern = 'pretrained_model.base_model.model.model.layers.{layer}'  # XXX: this works for peft LoRA, not tried for other things
        )
    else:
        reference_model = None

    if language_aspect_class.uses_local_neural_overseer:
        if neural_overseer_model is None:
            neural_overseer: BaseModel = training_model.duplicate(reference_model)
        else:
            neural_overseer = neural_overseer_model.eval()
        language_aspect_class.recruit_neural_overseer(neural_overseer)    

    if task_class.uses_local_neural_assessor:
        if neural_assessor_model is None:
            neural_assessor: BaseModel = training_model.duplicate(reference_model)
        else:
            neural_assessor = neural_assessor_model.eval()
        task_class.recruit_neural_assessor(neural_assessor)   

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



def get_tokenizer(
    model_name: str
):

    """
    Get tokenizer and set pad token to be same as end-of-sequence token

    Parameters
    ----------
    model_name : str
        The model name - get string from huggingface

    Returns
    -------
    AutoTokenizer
        Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_optimizer(
    optimizer_config: OptimizerConfig,
    model: nn.Module
) -> t.optim.Optimizer:
    
    """
    Returns an instance of torch.optim.Adam configured to work with submitted model/with settings from optimizer_config

    Parameters
    ----------
    optimizer_config : OptimizerConfig
        Optimisation config
    model : nn.Module
        torch model that the optimizer will be used with

    Returns
    -------
    torch.optim.Adam
        Configured optimizer
    """

    if optimizer_config.optimizer_name == "adam":
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer_config.learning_rate,
        )
        return optimizer


def get_model(
    model_config: ModelConfig, 
    tokenizer: AutoTokenizer, 
    output_delimitation_tokens: Dict[str, str], 
    generation_config=GenerationConfig,
    device: str = 'cuda'
) -> BaseModel:
    
    """_summary_

    

    model_config: ModelConfig, 
    tokenizer: AutoTokenizer, 
    output_delimitation_tokens: Dict[str, str], 
    generation_config=GenerationConfig,
    device: str = 'cuda'

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    model_name = model_config.model_name
    print(f'GETTING: {model_name}')
    if model_name == "gpt2":
        return GPT2Model(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device)
    elif model_name == 'meta-llama/Llama-3.1-8B-Instruct':
        return Llama_31_8B_Instruct(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device)
    elif model_name == 'meta-llama/Llama-3.2-1B-Instruct':
        return Llama_32_1B_Instruct(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device)
    elif model_name == 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B':
        return DeepSeek_R1_1pt5B(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device)
    elif "deepseek-ai" in model_name:
        return DeepSeek_R1_1pt5B(model_config=model_config, tokenizer=tokenizer, output_delimitation_tokens=output_delimitation_tokens, generation_config=generation_config, device=device)


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

            batch_messages_without_cot = model_class.batchize_conversation(
                system_prompt, batch_prompt_datas.no_cot_prompts
            )

            batch_messages_with_cot = model_class.batchize_conversation(
                system_prompt, batch_prompt_datas.cot_prompts
            )
            raise NotImplementedError('Use BaseModel.full_generate!')

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

    
