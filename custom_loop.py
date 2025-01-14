
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    set_seed,
)

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from trl.core import LengthSampler


tqdm.pandas()



@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./gpt2",
        metadata={"help": "the path to save the model"},
    )



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=100,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)


def build_dataset(
    config, dataset_name="deepmind/aqua_rat", input_min_text_length=5, input_max_text_length=10,cot_strategy="solve it step by step"
):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")


    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        options = [str(option) for option in sample['options']]
        
        prompt = (
            f"Question: {sample['question']}\n"
            f"Options: {', '.join(options)}\n"
            f"Instruction: {cot_strategy}\n\n"
            "Answer:"
        )

        


        sample["input_ids"] = tokenizer.encode(prompt)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds

min_input_length = 30
max_input_length = 40
dataset = build_dataset(config, input_min_text_length=min_input_length, input_max_text_length=max_input_length)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

set_seed(config.seed)

model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
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
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)


generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score
    texts = batch["response"]
    print(texts)
    
    rewards = [torch.tensor(100.0)]*len(texts)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)