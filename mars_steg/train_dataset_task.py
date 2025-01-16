from mars_steg.utils.trl import ScriptArguments, GenerationKwargs

import warnings

import torch
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    set_seed,
)

from trl import PPOConfig, PPOTrainer, create_reference_model


from mars_steg.task.tasks.math_dataset_task import MATH_Dataset_Task
from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
from mars_steg.utils.common import get_optimizer, get_tokenizer, get_model, parse_model_output, LLMTranscriptExtractionError


## Set-up before __name__ check in case we want to import this stuff from somewhere else


model_config, optimizer_config, train_config = ConfigLoader.load_config("mars_steg/configs/config.yaml")

config = PPOConfig(
    model_name=model_config.model_name,
    learning_rate=optimizer_config.learning_rate,
    log_with=train_config.log_with,
    ppo_epochs=train_config.ppo_epochs,
    mini_batch_size=train_config.mini_batch_size,
    batch_size=train_config.batch_size,
    gradient_accumulation_steps=train_config.gradient_accumulation_steps,
)

set_seed(config.seed)

# Get tokenizer
tokenizer = get_tokenizer(model_config.model_name)

# Get model
model = get_model(model_config, tokenizer)
ref_model = create_reference_model(model.model, num_shared_layers=3)

# Get optimizer
optimizer = get_optimizer(optimizer_config=optimizer_config, model=model)


# TODO: (low priority, after we get first script working) move this to config
dataset = 'mars_steg/dataset/MATH_dataset_minimal.csv'
math_task = MATH_Dataset_Task(dataset, EqualsSign())
train_dataset, test_dataset = math_task.test_train_split(train_proportion = 0.7)
generation_kwargs = GenerationKwargs.from_yaml("mars_steg/configs/generation_config/default.yaml").to_dict()
generation_kwargs["batch_size"] = train_config.batch_size


ppo_trainer = PPOTrainer(
    config,
    model.model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    optimizer=optimizer,
)

if __name__ == '__main__':



    # TODO: task objects should have a system_prompt attribute to give the system a prompt to use
    # TODO: get model-specific system prompt
    # -->  Connor
    messages = [model.get_message("system", train_dataset.system_prompt),]


    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1), 
        # we have to shuffle the data multiple times.
        # We should make sure this is reproducable
    # --> later
    

    for _ in range(train_config.num_train_epochs):

        batch_ticker = 1
        
        for prompt_data in tqdm(train_dataset.iterate_data(shuffle = True)):

            # TODO: message wrapping config --> Connor
            messages.append(model.get_message("user", prompt_data.cot_prompt))
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True, return_dict=True)

            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to("cuda") for key, tensor in inputs.items()}

            #Translate it into ppo.generate format which List[torch.tensors]
            query_tensors = []
            for input in inputs["input_ids"]:
                query_tensors.append(input)

            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            transcript_responses = ppo_trainer.generate(query_tensors, **generation_kwargs)
            decoded_responses = [tokenizer.decode(r.squeeze()) for r in transcript_responses]
            print(decoded_responses)

            # extraction of answer and cot (because cot_mode = True) parts of the transcript
            prompt_data.cot_transcript = decoded_responses[0]
            try:
                prompt_data.extracted_cot, prompt_data.extracted_final_answer_with_cot = parse_model_output(prompt_data.cot_transcript, True)
            except LLMTranscriptExtractionError as e:
                warnings.warn(f'Failed to parse: {e}\nSkipping example')

            composite_reward, task_score, language_score = train_dataset.reward_from_transcript(prompt_data)

            print(decoded_responses)

            print("--------")
            print(f"Rewarding with {composite_reward}")
            print(f"Task score: {task_score}")
            print(f"Language score: {language_score}")
            print("--------")

            #put reward in a list of tensor 
            if type(composite_reward) == list:
                rewards = []
                for reward in composite_reward:
                    rewards.append(torch.tensor(reward))
            else:
                rewards = [torch.tensor(composite_reward)]

            # TODO: put any batching back in here, e.g. by keeping a cache of 
            stats = ppo_trainer.step(query_tensors, transcript_responses, [torch.tensor(composite_reward)])
            ppo_trainer.log_stats(stats, inputs, composite_reward)
            
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

