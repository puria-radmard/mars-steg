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



def tokenize_example(x: dict, tokenizer, max_length=2560, cot=False) -> dict:
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
        resp_part = f"### RESPONSE:\n I will start reasoning process: "
    else:
        resp_part = f"### RESPONSE:\n ### ANSWER:\n {resp}"

    text = (
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"
        f"{instr_part}{context_part}{resp_part}"
    )
    return text


def tokenize_example(x: dict, max_length=2048, cot=False) -> dict:
    """Tokenize input examples with or without CoT."""
    instr = x["question"]
    resp = x["correct"]
    context = x.get("options", "")
    rationale = x.get("rationale", "")

    instr_part = f"### INSTRUCTION:\n{instr}"
    context_part = f"\nOPTIONS:\n{context}\n" if context else ""
    
    resp_part = f"### RESPONSE:\n I will start reasoning process: {rationale}\nTherefore response is "


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
        {"additional_special_tokens": ["### END\n ", "### INSTRUCTION:\n ", "### OPTIONS:\n ", "### ANSWER:\n ", "### RESPONSE:\n"]}
        )
    tokenizer.pad_token = tokenizer.eos_token
    

    checkpoint_name = "test-trainer-lab"
    local_checkpoint_path = "./experiments/test-trainer-lab_cot_2.8b/checkpoint-96500"

    from transformers import GPTNeoXForCausalLM, AutoTokenizer

    model = GPTNeoXForCausalLM.from_pretrained(model_name)

    # Resize the model embeddings to match the tokenizer's updated vocabulary size
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, local_checkpoint_path)

    
    # Move the model to the desired device (e.g., "cuda:1")
    model = model.to("cuda:0")

    test_ds = ds['test']
    for i in range(100):
       
        text  = tokenize_example(test_ds[i])

        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        tokens = model.generate(
            **inputs, 
            max_length=512,           
            num_beams=10,              # Number of beams for beam search (higher is more exhaustive)
            early_stopping=True,      # Stop as soon as the best beam is found
            pad_token_id=tokenizer.eos_token_id,  # Prevent padding token generation
            length_penalty=10.0,       
            temperature=2.0           # Keep temperature for controlling randomness
        )
        tokens = tokens.to("cpu")

        # Decode the generated tokens
        output = tokenizer.decode(tokens[0], skip_special_tokens=False)  # Skip special tokens for cleaner output

        # Print or return the result
        end_index = output.find("### END")
        output = output[:end_index]
        print(output)
        print(f"Real answer is: {test_ds[i]['correct']}")
