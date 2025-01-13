
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    PPOConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

import os
from custom_ppo_trainer import CustomPPOTrainer as PPOTrainer


from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
num_cores = 1
if torch.cuda.is_available():
    name_device = 'cuda'
    num_workers = num_cores
elif torch.backends.mps.is_available():
    name_device = 'mps'
    num_workers = num_cores
else:
    name_device = 'cpu'
    num_workers = 0


# def prepare_dataset(dataset, tokenizer):
#         """pre-tokenize the dataset before training; only collate during training"""

#         def tokenize(element):
#             outputs = tokenizer(
#                 element[dataset_text_field],
#                 truncation=True,
#                 max_length=100
#             )

#             return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

#         return dataset.map(
#             tokenize,
#             batched=True,
#             remove_columns=dataset.column_names,
#             num_proc=training_args.dataset_num_proc,
#         )

def prepare_dataset(dataset, tokenizer, max_length=128):
    """
    Pre-tokenize the dataset before training; only collate during training.
    This version prepares the dataset for RLHF by tokenizing input text.
    """
    def tokenize(element):
        outputs = tokenizer(
            element[dataset_text_field],  # Specify the text field containing input data
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Ensures consistent input size
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,  # Remove raw columns after tokenization
        num_proc=training_args.dataset_num_proc,  # Leverage multiple processes for speed
    )



"""
Take a look on run.sh code

python -i ppo_trainer_v3.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir models/minimal/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)
    quantization = 0 # Quantization flag

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    if quantization:
        quantization_config = get_quantization_config(model_args)
        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
    else:       
        model_kwargs = dict(
            revision=model_args.model_revision,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )


    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100

    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    


    dataset_text_field = "prompt"



    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=num_workers
        )

    def compute_reward(inputs, outputs):
        # Implement custom logic to compute reward based on inputs and outputs
        # For example, comparing outputs to high-quality references or using a classifier
        return torch.tensor([1.0] * len(inputs)).to(outputs.device)  # Placeholder reward

    ################
    # Training
    ################

    
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    num_epochs = 3


    for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            for batch in data_loader:
                # Convert input_ids and attention_mask to PyTorch tensors
                input_ids = torch.as_tensor(batch['input_ids']).to(name_device)
                attention_mask = torch.as_tensor(batch['attention_mask']).to(name_device)

                # Generate responses
                outputs = policy.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10000,
                    pad_token_id=tokenizer.pad_token_id,
                )

                # Compute rewards
                rewards = compute_reward(input_ids, outputs)

                # Run PPO step
                stats = trainer.step(input_ids, outputs, rewards)

                # Log stats
                print(stats)

            # Save the model checkpoint after each epoch
            policy.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")
            tokenizer.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")
