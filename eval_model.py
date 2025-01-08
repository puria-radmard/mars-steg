from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
)
import argparse
import os
from evaluate import load
import torch 
from peft import PeftModel, PeftConfig
import time


def tokenize_example_no_cot(x: dict, max_length=1024, cot=True) -> dict:
    """Tokenize input examples with or without CoT."""
    instr = x["question"]
    resp = x["correct"]
    context = x.get("options", "")
    rationale = x.get("rationale", "")
    context = str(context)
    context = context.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n")

    instr_part = f"### INSTRUCTION:\n{instr}"
    context_part = f"\n### OPTIONS:\n {context}\n" if context else ""

    resp_part = f"### RESPONSE:\n"

    
    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{instr_part}{context_part}{resp_part}"
    )
    print(text)


    return text


def tokenize_example_cot(x: dict, max_length=128, cot=False) -> dict:
    """Tokenize input examples with or without CoT."""
    instr = x["question"]
    resp = x["correct"]
    context = x.get("options", "")
    rationale = x.get("rationale", "")
    context = str(context)
    context = context.replace("[", "").replace("]", "").replace("'", "").replace(",", "\n")

    instr_part = f"<instruction> {instr}"
    context_part = f"\n <options> \n {context}\n" if context else ""

    resp_part = f"<answer>\n Lets think step by step: <cot>"

    
    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{instr_part}{context_part}{resp_part}"
    )

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COT GAP EXP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-m", "--model", type=str, help="model")
    parser.add_argument("-d", "--dataset", type=str, help="dataset")
    args = parser.parse_args()
    config = vars(args)

    dataset_name = args.dataset
    model_name = args.model
    local_training_root = "./experiments"

    cot = 1
    print("-------------------------------------")
    print("Running experiment with the following")
    print(dataset_name)
    print(model_name)
    print("-------------------------------------")

    ds = load_dataset(dataset_name, "raw")

    # Load the tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<end>", "<instruction>","<answer>",  "<options>", "<finalanswer>","<cot>","<endcot>"]}
        )
    tokenizer.pad_token = tokenizer.eos_token
    

    checkpoint_name = "test-trainer-lab"
    local_checkpoint_path = ""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize the model embeddings to match the tokenizer's updated vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, local_checkpoint_path)

    
    model = model.to("cuda:0")

    test_ds = ds['test']
    total_tokens = 0
    total_time = 0.0
    correct_answers = 0
    total_answers = 0
    for i in range(100):
        if not cot:
            text = tokenize_example_no_cot(test_ds[i])
        else: 
            text = tokenize_example_cot(test_ds[i])
        print("----------------")
        print(text)
        print("Input")
        print("----------------")
        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        tokens = model.generate(
            **inputs,
            max_length=512,  # Maximum token length of the output, useful for long COT reasoning steps.
            early_stopping=True,  # Stop early when the best beam is found to save time and computation.
            pad_token_id=tokenizer.eos_token_id,  # Ensure the model doesn't generate padding tokens.

        )
        tokens = tokens.to("cpu")

        # Decode the generated tokens
        output = tokenizer.decode(tokens[0], skip_special_tokens=False)  # Skip special tokens for cleaner output
        print(output)
        # Print or return the result
        end_index = output.find("<end>")
        Begginning_index = output.find("<finalanswer>")
        output = output[:end_index]
        print(output)
        
        
        answer = output[Begginning_index:end_index]
        
        extract = answer.split(" ")
        sol_llm = extract[-1]
        total_answers += 1
        real_sol = test_ds[i]['correct']
        print(f"Real answer is: {real_sol}")
        print(f"LLM answer is: {sol_llm}")
        if len(sol_llm) >= 1:
            if sol_llm[0] == real_sol:
                correct_answers += 1
                print({correct_answers/total_answers})
       
