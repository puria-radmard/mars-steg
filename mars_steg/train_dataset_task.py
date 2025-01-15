from mars_steg.utils.trl import ScriptArguments, GenerationKwargs

import torch
from datasets import load_dataset
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



## Set-up before __name__ check in case we want to import this stuff from somewhere else

parser = HfArgumentParser(ScriptArguments)
script_args, _, _ = parser.parse_args_into_dataclasses()

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)
set_seed(config.seed)

model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16,quantization_config=quantization_config,)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
ref_model = create_reference_model(model, num_shared_layers=3)

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token



ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
    )



if __name__ == '__main__':


    # TODO: Train-test split here!
    train_dataset = MATH_Dataset_Task()
    test_dataset = MATH_Dataset_Task()
    generation_kwargs = GenerationKwargs.from_yaml("generation_kwargs.yaml").to_dict()
    generation_kwargs["batch_size"] = script_args.batch_size

    #TODO datasets obejcts shoudl have a system_prompt attribute to give the system a prompt to use
    messages = [
        {"role": "system", "content": f"{train_dataset.system_prompt}"},
    ]

    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1), 
        # we have to shuffle the data multiple times.
        # We should make sure this is reproducable
    

    for _ in range(script_args.num_train_epochs):

        batch_ticker = 1
        
        # TODO: incorporate tqdm here
        for prompt_data in tqdm(train_dataset.iterate_data(shuffle = True)):

            # TODO: Puria has no idea how this works
            messages.append({"role": "user", "content": prompt_data.cot_prompt})
            formatted_chat = tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True, return_dict=True)
            # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
            inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            print("Tokenized inputs:\n", inputs)
            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            # TODO: We are replacing `*query`` here. This will be a combination of prompt_data.cot_prompt and any other data that is needed (put this in prompt_data.info)
            transcript_responses = ppo_trainer.generate(inputs["input_ids"], **generation_kwargs)
            decoded_responses = [tokenizer.decode(r.squeeze()) for r in transcript_responses]

            # TODO: some tokenisation of the response here! i.e. to replicate `batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]`
            # TODO: extraction of answer and cot (because cot_mode = True) parts of the transcript
            prompt_data.register_transcript(transcript_responses, cot_mode = True)

            composite_reward, task_score, language_score = train_dataset.reward_from_transcript(prompt_data)

            print(decoded_responses)

            print("--------")
            print(f"Rewarding with {composite_reward}")
            print(f"Task score: {task_score}")
            print(f"Language score: {language_score}")
            print("--------")

            # TODO: put any batching back in here, e.g. by keeping a cache of 
            stats = ppo_trainer.step(inputs["input_ids"], [prompt_data.cot_transcript], [composite_reward])
            ppo_trainer.log_stats(stats, inputs, composite_reward)
            
            # Save model every 100 batchs
            if batch_ticker % 100 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(script_args.model_save_path)
            batch_ticker += 1


                # ### Keep an eye on whether no CoT performance is also going up - this might mean the model is just getting better at the tasks
                # # and ignoring its CoT altogether
                # with torch.no_grad():
                #     ...
                #     # XXX: use test_dataset here
                #     ...

