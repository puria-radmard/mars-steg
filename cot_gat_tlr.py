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




def tokenize_example(x: dict, tokenizer, max_length=1024, cot=True) -> dict:
    """Tokenize input examples with or without CoT."""
    instr = x["question"]
    resp = x["correct"]
    context = x.get("options", "")
    rationale = x.get("rationale", "")
    context = str(context)
    context = context.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n")

    instr_part = f"### INSTRUCTION:\n{instr}"
    context_part = f"\n### OPTIONS:\n {context}\n" if context else ""
    if cot and rationale:
        resp_part = f"### RESPONSE:\n I will start reasoning process: {rationale}\n### ANSWER:\n  Therefore response is {resp}"
    else:
        resp_part = f"### RESPONSE:\n### ANSWER: {resp}"

    
    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{instr_part}{context_part}{resp_part}\n### END\n"
    )
    print(text)

    return tokenizer(text, max_length=max_length, truncation=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COT GAP EXP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name or path")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset name or path")
    args = parser.parse_args()

    # Configuration
    dataset_name = args.dataset
    model_name = args.model
    local_training_root = "./experiments"
    checkpoint_name = "test-trainer-lab_cot_1b_no_cot"
    local_checkpoint_path = os.path.join(local_training_root, checkpoint_name)

    # Load Dataset
    ds = load_dataset(dataset_name, "raw")
    cot = False  # Change this flag to toggle CoT mode
    lora = False
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["### END\n ", "### INSTRUCTION:\n ", "### OPTIONS:\n ", "### ANSWER:\n ", "### RESPONSE:\n"]}
    )

    # Tokenize Dataset
    tokenize_fn = lambda x: tokenize_example(x, tokenizer, cot=cot)
    remove_columns = [] if cot else ["rationale"]
    tokenized_dataset = ds.map(tokenize_fn, remove_columns=remove_columns)

    # Split Dataset
    TRAINING_SIZE = 90000
    SEED = 42
    split_dataset = tokenized_dataset["train"].train_test_split(train_size=TRAINING_SIZE, test_size=100, seed=SEED)

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
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            **lora_params
        )
        model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()
                

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=local_checkpoint_path,
        num_train_epochs=10,
        per_device_train_batch_size=4,
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


    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
    )


    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_state()

    eval_results = trainer.evaluate()
    print(eval_results)
