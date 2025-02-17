"""
Module Name: cot_evaluation.py
------------------------


Description:
------------

This file is used to evaluate the CoT gap of a model. The CoT gap is the difference in reward between a model that uses the CoT and a model that does not use the CoT. 
The CoT gap is a measure of the effectiveness of the CoT in improving the model's performance. The CoT gap is calculated by comparing the reward obtained by the model with the CoT to the reward obtained by the model without the CoT.
The CoT gap is calculated for a given dataset and is used to evaluate the performance of the model.

See Also:
---------
- mars_steg/utils.common.evaluate_cot_gap_summary

"""


from __future__ import annotations

import wandb
from dataclasses import asdict
import sys
import torch

from transformers import set_seed

from trl import PPOConfig, PPOTrainer

from mars_steg.config import ConfigLoader, ExperimentArgs, PromptConfig
from mars_steg.utils.common import (
    get_dataloaders_and_ref_model,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_device,
    evaluate_cot_gap_summary,
)


print(sys.argv)
experiment_args_path = sys.argv[1]
config_path = sys.argv[2]   # e.g. "mars_steg/configs/config.yaml"
prompt_config_path = sys.argv[3]   # e.g. "mars_steg/configs/prompt/math_training_deepseek.yaml"


device = get_device()

with torch.no_grad():

    model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(config_path)

    prompt_config = PromptConfig.from_file(prompt_config_path)

    output_delimitation_tokens = {
    "start_scratchpad_token": prompt_config.start_scratchpad_token,
    "end_scratchpad_token": prompt_config.end_scratchpad_token,
    "start_output_token": prompt_config.start_output_token,
    "end_output_token": prompt_config.end_output_token,
}

    generation_config.batch_size = train_config.batch_size

    # PPO Configuration
    config = PPOConfig(
        model_name=model_config.model_name,
        learning_rate=optimizer_config.learning_rate,
        log_with=train_config.log_with,
        ppo_epochs=train_config.ppo_epochs,
        mini_batch_size=train_config.mini_batch_size,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )


    experiment_args = ExperimentArgs.from_yaml(experiment_args_path)

    wandb.init(
        project="MARS-STEGO",
        config={
            'ppo_config': config,
            'model_config': model_config,
            'optimizer_config': optimizer_config,
            'train_config': train_config,
            'experiment_args': experiment_args,
        }
    )

    # Set random seed for reproducibility
    set_seed(config.seed)


    tokenizer = get_tokenizer(model_config.model_name)
    generation_config.pad_token_id = tokenizer.eos_token_id 
    model = get_model(
        model_config, 
        tokenizer, 
        output_delimitation_tokens=output_delimitation_tokens, 
        generation_config=generation_config,
        device=device)
    optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

    generation_kwargs = generation_config.to_dict()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, ref_model = get_dataloaders_and_ref_model(
        dataset_class_name = experiment_args.dataset_class_name,
        penalisation_class_name = experiment_args.penalisation_class_name,
        prompt_config = prompt_config,
        dataset_class_kwargs = experiment_args.dataset_class_kwargs,
        penalisation_class_kwargs = experiment_args.penalisation_class_kwargs,
        train_proportion = train_config.train_proportion,
        validation_proportion = train_config.validation_proportion,
        batch_size = train_config.batch_size,
        training_model = model,
        override_create_ref_model = train_config.create_ref_model,
        number_shared_layers_for_ref_model = train_config.number_shared_layers_for_ref_model
    )

    ppo_trainer = PPOTrainer(
        config,
        model.model,
        ref_model=ref_model if train_config.create_ref_model else None,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )


    cot_gap = evaluate_cot_gap_summary(
        ppo_trainer = ppo_trainer, 
        tokenizer = tokenizer, 
        dataloader = train_loader, 
        system_prompt = val_dataset.system_prompt,
        model_class = model, 
        device = device, 
        generation_kwargs = generation_kwargs,
        output_delimitation_tokens= output_delimitation_tokens,
        skipping_failed_parsing_examples= train_config.skipping_failed_parsing_examples
    )
    print(cot_gap.summary)
    wandb.log(asdict(cot_gap))

