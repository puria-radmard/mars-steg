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
from transformers import DataCollatorWithPadding

from trl import (
    ModelConfig,
    PPOConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.ppo_trainer import PPOTrainer


class RewardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs):
        return torch.tensor([1.0] * len(inputs)).to(outputs.device)  # Placeholder reward

# Device setup
num_cores = 12
if torch.cuda.is_available():
    name_device = 'cuda'
    num_workers = num_cores
elif torch.backends.mps.is_available():
    name_device = 'mps'
    num_workers = num_cores
else:
    name_device = 'cpu'
    num_workers = 0

def prepare_dataset(dataset, tokenizer, max_length=128, cot_strategy="solve it step by step"):
    def tokenize(element):
        options = [str(option) for option in element['options']]
        
        prompt = (
            f"Question: {element['question']}\n"
            f"Options: {', '.join(options)}\n"
            f"Instruction: {cot_strategy}\n\n"
            "Answer:"
        )
        
        outputs = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=False,
        remove_columns=["question", "options", "rationale", "correct"],  # Remove unnecessary fields
    )

    return tokenized_dataset







if __name__ == "__main__":
    # Parse arguments
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Remove output directory if it exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # Quantization flag
    quantization = 0

    # Model & Tokenizer setup
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
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
    )
    reward_model = RewardModel()
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
    )

    peft_config = get_peft_config(model_args)
    ref_policy = None
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code
        )

    # Load and prepare dataset
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    number_of_samples = 100 #len(dataset)-eval_samples
    train_dataset = dataset.select(range(min(len(dataset) - eval_samples,number_of_samples)))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))

    tokenized_train_dataset = prepare_dataset(train_dataset, tokenizer)
    tokenized_eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    print(tokenized_train_dataset[0])

    # Training setup
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        value_model=value_model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        peft_config=peft_config,
        reward_model=reward_model,
        data_collator=data_collator,
    )

    trainer.train()