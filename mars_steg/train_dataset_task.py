from __future__ import annotations

import warnings
import numpy as np
import wandb
import sys

import torch
from tqdm import tqdm
from transformers import set_seed

from trl import PPOConfig, PPOTrainer, create_reference_model

from mars_steg.language.language_aspects.neural_overseer import NeuralOverseer
from mars_steg.utils.prompt_data import BatchPromptData
from mars_steg.utils.exceptions import LLMTranscriptExtractionError
from mars_steg.config import ConfigLoader, ExperimentArgs
from mars_steg.utils.common import (
    get_dataloaders,
    batchize_conversation,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_device,
    get_rewards_and_training_datas,

    evaluate_cot_gap_and_ratio,
)






device = get_device()


model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(
    "mars_steg/configs/config.yaml"
)

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


experiment_args = ExperimentArgs.from_yaml(sys.argv[1])


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
model = get_model(model_config, tokenizer)
optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

generation_kwargs = generation_config.to_dict()

if train_config.create_ref_model is True:
    ref_model = create_reference_model(
        model.model, num_shared_layers=train_config.number_shared_layers_for_ref_model
    )


train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_dataloaders(
    dataset_class_name = experiment_args.dataset_class_name,
    penalisation_class_name = experiment_args.penalisation_class_name,
    dataset_class_kwargs = experiment_args.dataset_class_kwargs,
    penalisation_class_kwargs = experiment_args.penalisation_class_kwargs,
    train_proportion = train_config.train_proportion,
    validation_proportion = train_config.validation_proportion,
    batch_size = train_config.batch_size,
    reference_model = ref_model,        # This may be needed if a neural language aspect is used!
)







ppo_trainer = PPOTrainer(
    config,
    model.model,
    ref_model=ref_model if train_config.create_ref_model else None,
    tokenizer=tokenizer,
    optimizer=optimizer,
)

output_delimitation_tokens = {
    #"start_scratchpad_token": "<SCRATCHPAD_REASONING>",
    #"end_scratchpad_token": "</SCRATCHPAD_REASONING>",
    "start_output_token": "Answer:",
    #"end_output_token": "</OUTPUT>",
}

if __name__ == "__main__":

    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1),
    # we have to shuffle the data multiple times.
    # We should make sure this is reproducable
    # --> later
    epochs_to_evaluate_list = np.linspace(0, train_config.num_train_epochs-1, train_config.number_of_evaluations)
    epochs_to_evaluate_list = [int(epoch) for epoch in epochs_to_evaluate_list]
    epoch_to_evaluate = epochs_to_evaluate_list.pop(0)

    for epoch in range(train_config.num_train_epochs):

        batch_ticker = 0

        if train_config.evaluate_cot_gap and epoch_to_evaluate == epoch:


            print("--------")
            print("CoT Gap Evaluation")
            print("--------")

            print("--------")
            print("Generating responses for CoT gap evaluation")
            print("--------")

            if train_config.cot_gap_evaluation_number_batches_samples is None:
                train_config.cot_gap_evaluation_number_batches_samples = len(val_loader)

            cot_gap, cot_ratio = evaluate_cot_gap_and_ratio(
                ppo_trainer = ppo_trainer, 
                tokenizer = tokenizer, 
                dataloader = val_loader, 
                system_prompt = val_dataset.system_prompt,
                model_class = model, 
                device = device, 
                generation_kwargs = generation_kwargs,
                output_delimitation_tokens= output_delimitation_tokens,
                number_of_batches = train_config.cot_gap_evaluation_number_batches_samples
            )

            print("--------")
            print("Logging CoT Gap")
            print("--------")

            wandb.log({"CoT Gap": cot_gap, "CoT Ratio": cot_ratio})

            if len(epochs_to_evaluate_list) > 0:
                epoch_to_evaluate = epochs_to_evaluate_list.pop(0)


        for batch_prompt_datas in tqdm(train_loader):

            batch_prompt_datas: BatchPromptData

            batch_messages = batchize_conversation(
                system_prompt=train_dataset.system_prompt,
                user_prompts= batch_prompt_datas.cot_prompts, 
                model_class= model
            )
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
            print("RESPONSES")
            for response in decoded_responses:
                print(response)
            print("-------------")
            # extraction of answer and cot (because cot_mode = True) parts of the transcript
            batch_prompt_datas.cot_transcripts = decoded_responses
            num_examples_failed = 0

            if isinstance(train_dataset.language_aspect, NeuralOverseer):
                generated_overseer_answers = train_dataset.language_aspect.get_overseer_generated_answers(batch_prompt_datas)
            else:
                generated_overseer_answers = None

            composite_reward_list, task_score_list, language_score_list, train_query_tensors, train_transcript_responses, num_examples_failed = get_rewards_and_training_datas(
                batch_prompt_datas=batch_prompt_datas,
                query_tensors=query_tensors,
                transcript_responses=transcript_responses, 
                train_dataset=train_dataset, 
                output_delimitation_tokens=output_delimitation_tokens,
                batch_overseer_answers=generated_overseer_answers
            )

            if num_examples_failed == len(batch_prompt_datas):
                optimize = False
                warnings.warn("All examples failed to parse. Not optimizing model")
                
            elif num_examples_failed != 0:
                warnings.warn(
                    f"Failed to parse {num_examples_failed} examples out of {len(batch_prompt_datas)}"
                )
                warnings.warn("Optimizing model with the rest of the examples (putting 0 reward for the failed examples)")
                optimize = True
            else:
                print("All examples parsed successfully")
                optimize = True

            print("--------")
            print(f"Rewarding example with {composite_reward_list}")
            print(f"Task score: {task_score_list}")
            print(f"Language score: {language_score_list}")
            print("--------")

            print("--------")
            print("Rewarding model")
            print("--------")

            if optimize:

                print("--------")
                print("Optimizing model")
                print("--------")

                stats = ppo_trainer.step(
                    train_query_tensors, train_transcript_responses, composite_reward_list
                )

                print("--------")
                print("Logging stats")
                print("--------")
                ppo_trainer.log_stats(
                    stats,
                    inputs,
                    [composite_reward_list, task_score_list, language_score_list],
                )
                wandb.log(stats)

            torch.cuda.empty_cache() 
            # Save model every 100 batchs
            if batch_ticker % train_config.save_frequency == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(model_config.model_save_path)

            batch_ticker += 1
            

            # ### Keep an eye on whether no CoT performance is also going up - this might mean the model is just getting better at the tasks
            # # and ignoring its CoT altogether
            # with torch.no_grad():
            #     ...
            #     # XXX: use test_dataset here
            #     ...
