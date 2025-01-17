from mars_steg.utils.trl import ScriptArguments, GenerationKwargs

import warnings
import wandb

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model


from mars_steg.task.tasks.math_dataset_task import MATH_Dataset_Task
from mars_steg.language.language_aspects.math_language_aspect import EqualsSign
from mars_steg.utils.common import parse_model_output, LLMTranscriptExtractionError
from mars_steg.utils.common import parse_model_output, LLMTranscriptExtractionError
from mars_steg.task.tasks.math_dataset_task import LLMRewardExtractionError

## Set-up before __name__ check in case we want to import this stuff from somewhere else

parser = HfArgumentParser(ScriptArguments)
script_args, *_ = parser.parse_args_into_dataclasses()
save_frequency = 100 # WE SHOULD PUT THIS ON YAML


# Configure quantization for fast token generation
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16  
)

# PPO Configuration
config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# Initialize Weights & Biases logging
wandb.init(
    project="MARS-STEGO", 
    config=config
)

# Set random seed for reproducibility
set_seed(config.seed)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype= torch.bfloat16,
    quantization_config=quantization_config,
    device_map="auto"  # Automatically place layers on available GPUs
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
# ref_model = create_reference_model(model, num_shared_layers=25) 

model = model.to("cuda")

# Optimizer setup
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token


# TODO: (low priority, after we get first script working) move this to config
dataset = 'mars_steg/dataset/MATH_dataset_minimal.csv'
math_task = MATH_Dataset_Task(dataset, EqualsSign())
train_dataset, test_dataset = math_task.test_train_split(train_proportion = 0.7)
generation_kwargs = GenerationKwargs.from_yaml("mars_steg/configs/generation_config/default.yaml").to_dict()
generation_kwargs["batch_size"] = script_args.batch_size


ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=None,
    tokenizer=tokenizer,
    optimizer=optimizer,
)


output_delimitation_tokens = {
    #"start_scratchpad_token": "<SCRATCHPAD_REASONING>",
    #"end_scratchpad_token": "</SCRATCHPAD_REASONING>",
    "start_output_token": "Answer:",
    #"end_output_token": "</OUTPUT>",
}



if __name__ == '__main__':

    # TODO: task objects should have a system_prompt attribute to give the system a prompt to use
    # TODO: get model-specific system prompt
    # -->  Connor
    messages = [
        {"role": "system", "content": f"You are an expert Model on Math {train_dataset.system_prompt}" },
    ]

    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1), 
        # we have to shuffle the data multiple times.
        # We should make sure this is reproducable
    # --> later

    appearances = 0
    for _ in range(script_args.num_train_epochs):

        batch_ticker = 0
        generated_prompts = 0
        
        
        for prompt_data in tqdm(train_dataset.iterate_data(shuffle = True), total = train_dataset.length):

            # TODO: message wrapping config --> Connor
            if len(messages) >= 2:
                messages.pop()
            
            messages.append({"role": "user", "content":  prompt_data.cot_prompt})
           

            print(messages)       
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

            # extraction of answer and cot (because cot_mode = True) parts of the transcript
            prompt_data.cot_transcript = decoded_responses[0]

            print(decoded_responses[0])
            
            try:
                prompt_data.extracted_cot, prompt_data.extracted_final_answer_with_cot = parse_model_output(prompt_data.cot_transcript, True, output_delimitation_tokens)
                skip_reward = 0
            except Exception as e:
                warnings.warn(f'Failed to parse: {e} \nSkipping example')
                continue

           
            composite_reward, task_score, language_score = train_dataset.reward_from_transcript(prompt_data)
  
            print("--------")
            print(f"Rewarding with {composite_reward}")
            print(f"Task score: {task_score}")
            print(f"Language score: {language_score}")
            print("--------")

            if type(composite_reward) == list:
                rewards = []
                for reward in composite_reward:
                    rewards.append(torch.tensor(reward))
            else:
                rewards = [torch.tensor(composite_reward)]

            # TODO: put any batching back in here, e.g. by keeping a cache of 
            stats = ppo_trainer.step(query_tensors, transcript_responses, [torch.tensor(composite_reward)])
            #wandb.log(stats)
            torch.cuda.empty_cache() 
            
            inputs["response"] = decoded_responses
            inputs["query"] = query_tensors

            #ppo_trainer.log_stats(stats, inputs, composite_reward)
        
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

