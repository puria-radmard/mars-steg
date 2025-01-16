import numpy as np
from mars_steg.utils.trl import ScriptArguments, GenerationKwargs

import warnings
import os

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    # BitsAndBytesConfig
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from torch.utils.data import DataLoader


from mars_steg.task.tasks.math_dataset_task import MATH_Dataset_Task, MATH_Torch_Dataset_Task
from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
from mars_steg.utils.common import parse_model_output, LLMTranscriptExtractionError

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
    
device = get_device()

# Optionally load configuration from YAML
config_path = os.path.join(os.path.abspath(""),"mars_steg", "configs", "training.yaml") 
print(config_path) # Replace with the actual path to your YAML file
try:
    script_args = ScriptArguments.from_yaml(config_path)
    print(f"Loaded configuration from {config_path}")
except FileNotFoundError or IsADirectoryError:
    print(f"Config file {config_path} not found. Using default.")


save_frequency = script_args.save_frequency
print(script_args)

# quantization_config = BitsAndBytesConfig(load_in_4bit=True)    

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)
set_seed(config.seed)

# model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16,quantization_config=quantization_config,)
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16).to(device)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
#TODO create a boolean to check if we want to use the ref model
ref_model = create_reference_model(model, num_shared_layers=script_args.number_shared_layers_for_ref_model)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token


# TODO: (low priority, after we get first script working) move this to config
dataset = 'mars_steg/dataset/MATH_dataset_minimal.csv'
math_task = MATH_Torch_Dataset_Task(dataset, EqualsSign())
train_dataset, test_dataset = math_task.test_train_split(train_proportion = script_args.train_proportion)
train_loader, test_loader = DataLoader(train_dataset, batch_size=script_args.batch_size, shuffle=True, collate_fn=math_task.prompt_data_collate_fn), DataLoader(test_dataset, batch_size=script_args.batch_size, shuffle=True, collate_fn=math_task.prompt_data_collate_fn)
generation_kwargs = GenerationKwargs.from_yaml("mars_steg/configs/generation_config/default.yaml").to_dict()
generation_kwargs["batch_size"] = script_args.batch_size


ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    optimizer=optimizer,
)

def batchize_conversation(system_prompt, user_prompts):
    batch_messages = []
    for user_prompt in user_prompts:
        batch_messages.append([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return batch_messages



if __name__ == '__main__':

    # TODO: task objects should have a system_prompt attribute to give the system a prompt to use
    # TODO: get model-specific system prompt
    # -->  Connor
    messages = [
        {"role": "system", "content": f"{train_dataset.system_prompt}"},
    ]


    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1), 
        # we have to shuffle the data multiple times.
        # We should make sure this is reproducable
    # --> later
    

    for _ in range(script_args.num_train_epochs):

        batch_ticker = 1
        
        for batch_prompt_datas in tqdm(train_loader):

            # TODO: message wrapping config --> Connor
            batch_messages = batchize_conversation(train_dataset.system_prompt, batch_prompt_datas.cot_prompts)
            inputs = tokenizer.apply_chat_template(batch_messages, return_tensors="pt",add_generation_prompt=True, return_dict=True, padding=True)

            # Move the tokenized inputs to the same device the model is on (GPU/CPU)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

            #Translate it into ppo.generate format which List[torch.tensors]
            query_tensors = []
            for input in inputs["input_ids"]:
                query_tensors.append(input)

            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            transcript_responses = ppo_trainer.generate(query_tensors, **generation_kwargs)
            decoded_responses = [tokenizer.decode(r.squeeze()) for r in transcript_responses]

            # extraction of answer and cot (because cot_mode = True) parts of the transcript
            batch_prompt_datas.cot_transcripts = decoded_responses
            composite_reward_list, task_score_list, language_score_list = [], [], []
            train_query_tensors = []
            train_transcript_responses = []
            error = 0
            for i, prompt_data in tqdm(enumerate(batch_prompt_datas)):
                try:
                    print("Normal hello")
                    prompt_data.extracted_cot, prompt_data.extracted_final_answer_with_cot = parse_model_output(prompt_data.cot_transcript, True)
                    composite_reward, task_score, language_score = train_dataset.reward_from_transcript(prompt_data)
                    composite_reward_list.append(torch.tensor(composite_reward))
                    task_score_list.append(task_score)
                    language_score_list.append(language_score)
                    train_query_tensors.append(query_tensors[i])
                    train_transcript_responses.append(transcript_responses[i])
                    # composite_reward_list.append(torch.tensor(1.0))
                    # task_score_list.append(torch.tensor(1.0))
                    # language_score_list.append(torch.tensor(1.0))
                except LLMTranscriptExtractionError as e:
                    error +=1
                    warnings.warn(f'Failed to parse: {e}\nSkipping example')

            if error == len(batch_prompt_datas):
                raise ValueError("No examples were successfully parsed for reward function. Check the data and the parsing function.")
            elif error != 0:
                warnings.warn(f"Failed to parse {error} examples out of {len(batch_prompt_datas)}")
            else:
                print("All examples parsed successfully")


            # print("--------")
            # print(f"Rewarding example with {composite_reward_list[0]}")
            # print(f"Task score: {task_score_list[0]}")
            # print(f"Language score: {language_score_list[0]}")
            # print(f"For example: {train_query_tensors[0]}")
            # print("--------")

            # TODO: put any batching back in here, e.g. by keeping a cache of 
            stats = ppo_trainer.step(train_query_tensors, train_transcript_responses, composite_reward_list)
            # stats = {
            #     "loss": 0.234,  # Example loss
            #     "policy_loss": 0.112,  # Example policy loss
            #     "value_loss": 0.089,  # Example value loss
            #     "entropy": 0.015,  # Example entropy
            #     "reward_mean": torch.mean(torch.tensor([1.2, 0.8])).item(),  # Average reward
            #     "reward_std": torch.std(torch.tensor([1.2, 0.8])).item(),  # Reward standard deviation
            #     "kl_divergence": 0.02  # Example KL divergence
            # }
            ppo_trainer.log_stats(stats, inputs, [composite_reward_list, task_score_list, language_score_list])
            
            # Save model every 100 batchs
            if batch_ticker % save_frequency == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(script_args.model_save_path)

            batch_ticker += 1


                # ### Keep an eye on whether no CoT performance is also going up - this might mean the model is just getting better at the tasks
                # # and ignoring its CoT altogether
                # with torch.no_grad():
                #     ...
                #     # XXX: use test_dataset here
                #     ...

