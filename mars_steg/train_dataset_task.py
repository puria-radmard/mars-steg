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
script_args = parser.parse_args_into_dataclasses()[0]

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

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

tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token



ppo_trainer = PPOTrainer(
        config,
        model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        
        # XXX: TODO: make PPOTrainer accept this. Might have to make Task class inherit from torch Dataset, see e.g. https://pytorch.org/tutorials/beginner/data_loading_tutorial, 
        # 
        dataset=train_dataset,

        data_collator=collator,
        optimizer=optimizer,
    )



if __name__ == '__main__':

    # Initialise training
    ref_model = create_reference_model(model, num_shared_layers=3)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # TODO: Train-test split here!
    train_dataset = MATH_Dataset_Task()
    test_dataset = MATH_Dataset_Task()

    # TODO: pimp this class
    generation_kwargs = GenerationKwargs(30, 3, 0.3, True, tokenizer.eos_token_id, 512)


    # TODO (low prioirity): if we are doing more than 1 epoch (here epoch = repeat over all data, so probably not doing more than 1), 
        # we have to shuffle the data multiple times.
        # We should make sure this is reproducable
    num_epochs = 1

    for _ in range(num_epochs):

        batch_ticker = 1    # XXX: used to be called epochs
        
        # TODO: incorporate tqdm here
        # XXX: what do we lose by doing this instead of iterating over `ppo_trainer.dataloader`?
            # From `clean_loop_training.py` it seems like we were not actually batching response generation anyway, so this should be fine
        for prompt_data in train_dataset.iterate_data(shuffle = True):

            batch_ticker += 1

            # TODO: Puria has no idea how this works
            messages = [
                {"role": "system", "content": f"Your a helpful agent, {system_prompt}"},
                {"role": "user", "content": prompt_data.cot_prompt},
            ]
            query = tokenizer.apply_chat_template(messages, return_tensors="pt",add_generation_prompt=True, return_dict=True)
            
            # Generate the with-CoT transcript (no-CoT transcript not needed during training)
            # TODO: We are replacing `*query`` here. This will be a combination of prompt_data.cot_prompt and any other data that is needed (put this in prompt_data.info)
            transcript_response = ppo_trainer.generate(query, **generation_kwargs.__dict__())

            # TODO: some tokenisation of the response here! i.e. to replicate `batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]`
            # TODO: extraction of answer and cot (because cot_mode = True) parts of the transcript
            prompt_data.register_transcript(transcript_response, cot_mode = True)

            composite_reward, task_score, language_score = math_task.reward_from_transcript(prompt_data)

            print(text)

            print("--------")
            print(f"Rewarding with {penzalization}")
            print("--------")

            # TODO: put any batching back in here, e.g. by keeping a cache of 
            stats = ppo_trainer.step([query], [prompt_data.cot_transcript], [composite_reward])
            ppo_trainer.log_stats([stats], [batch], [rewards])
            
            # Save model every 100 batchs
            if batch_ticker % 100 == 0:
                if ppo_trainer.accelerator.is_main_process:
                    ppo_trainer.save_pretrained(model_save_path)


                ### Keep an eye on whether no CoT performance is also going up - this might mean the model is just getting better at the tasks
                # and ignoring its CoT altogether
                with torch.no_grad():
                    ...
                    # XXX: use test_dataset here
                    ...

