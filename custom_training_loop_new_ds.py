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


def build_dataset(tokenizer, dataset_name="deepmind/aqua_rat",  max_length=1000, cot_strategy="solve it step by step"
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    ds = load_dataset(dataset_name, split="train")

    def tokenize(element):
        # Prepare the prompt using the question and options
        options = [str(option) for option in element['options']]
        
        prompt = (
            f"Question: {element['question']}\n"
            f"Options: {', '.join(options)}\n"
            f"Instruction: {cot_strategy}\n\n"
            "Answer:"
        )
        
        # Tokenize the prompt
        outputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        return outputs
    
    # Apply the tokenization function to the dataset
    tokenized_dataset = ds.map(
        tokenize,
        batched=False,
        remove_columns=["question", "options", "rationale", "correct"],  # Remove unnecessary fields
    )

    tokenized_dataset.set_format(type="torch")

    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=False)["train"]

    return tokenized_dataset


tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
dataset = build_dataset(tokenizer=tokenizer)


def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

model = AutoModelForCausalLM.from_pretrained(config.model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

ref_model = create_reference_model(model, num_shared_layers=3)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
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
output_max_length = 50
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors_ids, query_tensors_masks = batch["input_ids"], batch["attention_mask"]

    # Get response from the policy model
    response_tensors = []
    for query_ids in query_tensors_ids:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query_ids, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    # Ensure token IDs are within the valid range before decoding
    valid_responses = []
    for r in response_tensors:
        valid_token_ids = [token_id for token_id in r.squeeze().tolist() if 0 <= token_id < tokenizer.vocab_size]
        if len(valid_token_ids) != len(r.squeeze().tolist()):
            print("Warning: Some token IDs are out of range and will be skipped.")
        valid_responses.append(tokenizer.decode(valid_token_ids))
    
    batch["response"] = valid_responses

    # Compute sentiment score
    texts = batch["response"]
    print(texts)
    
    rewards = [torch.tensor(1.0)]*len(texts)

    # Run PPO step
    stats = ppo_trainer.step(query_tensors_ids, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 100 == 0:
        if ppo_trainer.accelerator.is_main_process:
            ppo_trainer.save_pretrained(model_save_path)