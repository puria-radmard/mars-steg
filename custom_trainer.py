from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer
import argparse
import os
from peft import LoraConfig, get_peft_model, TaskType
import re
from transformers import Trainer
import torch
from transformers import PreTrainedTokenizerBase

def tokenize_example(x: dict, tokenizer, max_length=800, cot=True) -> dict:
    """Tokenize input examples with or without CoT."""
    instr = x["question"]
    resp = x["correct"]
    context = x.get("options", "")
    rationale = x.get("rationale", "")
    context = str(context)
    context = (
        context.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n")
    )

    instr_part = f"<instruction> {instr}"
    context_part = f"\n <options> \n {context}\n" if context else ""
    if cot and rationale:
        resp_part = f"<answer>\n Lets think step by step: <cot> {rationale} <endcot> \n <finalanswer> {resp}"

    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{instr_part}{context_part}{resp_part}\n<end>\n"
    )
    print(text)

    return tokenizer(text, max_length=max_length, truncation=True)


class CustomTrainer(Trainer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs):
        """
        Custom Trainer with tokenizer support.
        """
        super().__init__(*args, **kwargs)
        self._tokenizer = tokenizer  # Store the tokenizer for use

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Custom loss computation with decoding for analysis.
        """
        # Extract labels from inputs
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Compute the main cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        input_texts = self._tokenizer.batch_decode(
            inputs["input_ids"], skip_special_tokens=False
        )

        expected_answer = None
        for text in input_texts:

            match = re.search(r"<finalanswer>\s*([A-D])\s*<end>", text)
            if match:
                expected_answer = match.group(1)

        decoded_texts = self.decode_text(logits)

        penalty = 0.0
        for text in decoded_texts:

            match = re.search(r"<finalanswer>\s*([A-D])\s*<end>", text)
            if match:
                final_answer = match.group(1)

                # Apply penalty if the final answer does not match the expected answer
                if final_answer != expected_answer:
                    penalty += 1.0 
            else:

                penalty += 0.5 

        total_loss = loss + penalty

        return (total_loss, outputs) if return_outputs else total_loss

    def decode_text(self, logits_or_ids, skip_special_tokens=False):
        """
        Decode logits or token IDs into human-readable text using the tokenizer.
        """
        if len(logits_or_ids.shape) == 3:  # If logits are passed
            token_ids = torch.argmax(logits_or_ids, dim=-1)
        else:  # If input_ids are passed directly
            token_ids = logits_or_ids

        decoded_text = self._tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        return decoded_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COT GAP EXP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Dataset name or path"
    )
    args = parser.parse_args()

    # Configuration
    dataset_name = args.dataset
    model_name = args.model
    local_training_root = "./experiments"
    checkpoint_name = "test-trainer-lab_cot_2.8b_new_cot_reward"
    local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)

    # Load Dataset
    ds = load_dataset(dataset_name, "raw")
    cot = True  # Change this flag to toggle CoT mode
    lora = True
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<end>",
                "<instruction>",
                "<answer>",
                "<options>",
                "<finalanswer>",
                "<cot>",
                "<endcot>",
            ]
        }
    )

    # Tokenize Dataset
    tokenize_fn = lambda x: tokenize_example(x, tokenizer, cot=cot)
    remove_columns = [] if cot else ["rationale"]
    tokenized_dataset = ds.map(tokenize_fn, remove_columns=remove_columns)

    # Split Dataset
    TRAINING_SIZE = 90000
    SEED = 42
    split_dataset = tokenized_dataset["train"].train_test_split(
        train_size=TRAINING_SIZE, test_size=100, seed=SEED
    )

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    if lora:
        lora_params = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
        }
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_params)
        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=local_checkpoint_path,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        save_steps=500,
        save_total_limit=3,
        learning_rate=3e-4,
        logging_steps=1000,
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,  # Mixed precision
        gradient_accumulation_steps=4,
        report_to="none",
        optim="adamw_torch",
        logging_dir=os.path.join(local_checkpoint_path, "logs"),
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    trainer.save_state()

    eval_results = trainer.evaluate()
    print(eval_results)
