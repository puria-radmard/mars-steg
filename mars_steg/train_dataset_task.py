import numpy as np
from mars_steg.utils.trl import GenerationKwargs

import warnings
import wandb
import sys

import torch
from tqdm import tqdm
from transformers import set_seed

from trl import PPOConfig, PPOTrainer, create_reference_model

from mars_steg.utils.exceptions import LLMTranscriptExtractionError
from mars_steg.config import ConfigLoader, ExperimentArgs
from mars_steg.utils.common import (
    get_dataloaders,
    batchize_conversation,
    get_optimizer,
    get_tokenizer,
    get_model,
    get_device,
    parse_model_output,
    LLMTranscriptExtractionError,
)




device = get_device()


model_config, optimizer_config, train_config = ConfigLoader.load_config(
    "mars_steg/configs/config.yaml"
)


config = PPOConfig(
    model_name=model_config.model_name,
    learning_rate=optimizer_config.learning_rate,
    log_with=train_config.log_with,
    ppo_epochs=train_config.ppo_epochs,
    mini_batch_size=train_config.mini_batch_size,
    batch_size=train_config.batch_size,
    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
)


wandb.init(project="MARS-STEGO", config=config)

set_seed(config.seed)


tokenizer = get_tokenizer(model_config.model_name)
model = get_model(model_config, tokenizer)
optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

if train_config.create_ref_model is True:
    ref_model = create_reference_model(
        model, num_shared_layers=train_config.number_shared_layers_for_ref_model
    )

experiment_args = ExperimentArgs.from_yaml(sys.argv[1])


train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(
    dataset_class_name = experiment_args.dataset_class_name,
    penalisation_class_name = experiment_args.penalisation_class_name,
    dataset_class_kwargs = experiment_args.dataset_class_kwargs,
    penalisation_class_kwargs = experiment_args.penalisation_class_kwargs,
    train_proportion = train_config.train_proportion,
    batch_size = train_config.batch_size
)


generation_kwargs = GenerationKwargs.from_yaml(
    "mars_steg/configs/generation_config/default.yaml"
).to_dict()
generation_kwargs["batch_size"] = train_config.batch_size


ppo_trainer = PPOTrainer(
    config,
    model.model,
    ref_model=ref_model if train_config.create_ref_model else None,
    tokenizer=tokenizer,
    optimizer=optimizer,
)


if __name__ == "__main__":

    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1),
    # we have to shuffle the data multiple times.
    # We should make sure this is reproducable
    # --> later

    for _ in range(train_config.num_train_epochs):

        batch_ticker = 0

        for batch_prompt_datas in tqdm(train_loader):

            batch_messages = batchize_conversation(
                train_dataset.system_prompt, batch_prompt_datas.cot_prompts, model
            )
            inputs = tokenizer.apply_chat_template(
                batch_messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
                padding=True,
            )

            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            # Translate it into ppo.generate format which List[torch.tensors]
            query_tensors = []
            for input in inputs["input_ids"]:
                query_tensors.append(input)

            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            transcript_responses = ppo_trainer.generate(
                query_tensors, **generation_kwargs
            )
            decoded_responses = [
                tokenizer.decode(r.squeeze()) for r in transcript_responses
            ]

            # extraction of answer and cot (because cot_mode = True) parts of the transcript
            batch_prompt_datas.cot_transcripts = decoded_responses
            composite_reward_list, task_score_list, language_score_list = [], [], []
            train_query_tensors = []
            train_transcript_responses = []
            num_examples_failed = 0
            for i, prompt_data in tqdm(enumerate(batch_prompt_datas)):
                try:
                    (
                        prompt_data.extracted_cot,
                        prompt_data.extracted_final_answer_with_cot,
                    ) = parse_model_output(prompt_data.cot_transcript, True)
                    composite_reward, task_score, language_score = (
                        train_dataset.reward_from_transcript(prompt_data)
                    )
                    composite_reward_list.append(torch.tensor(composite_reward))
                    task_score_list.append(task_score)
                    language_score_list.append(language_score)
                    train_query_tensors.append(query_tensors[i])
                    train_transcript_responses.append(transcript_responses[i])
                except LLMTranscriptExtractionError as e:
                    num_examples_failed += 1
                    warnings.warn(f"Failed to parse: {e}\nSkipping example")

            if num_examples_failed == len(batch_prompt_datas):
                raise LLMTranscriptExtractionError(
                    "No examples were successfully parsed for reward function. Check the data and the parsing function."
                )
            elif num_examples_failed != 0:
                warnings.warn(
                    f"Failed to parse {num_examples_failed} examples out of {len(batch_prompt_datas)}"
                )
            else:
                print("All examples parsed successfully")

            print("--------")
            print(f"Rewarding example with {composite_reward_list}")
            print(f"Task score: {task_score_list}")
            print(f"Language score: {language_score_list}")
            print(f"For example: {train_query_tensors[0]}")
            print("--------")

            stats = ppo_trainer.step(
                train_query_tensors, train_transcript_responses, composite_reward_list
            )
            ppo_trainer.log_stats(
                stats,
                inputs,
                [composite_reward_list, task_score_list, language_score_list],
            )

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
