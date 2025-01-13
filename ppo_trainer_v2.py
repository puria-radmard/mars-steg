


from datasets import load_dataset
from trl import PPOConfig
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from tqdm import tqdm
import torch
from transformers import pipeline
import os

num_cores = os.cpu_count()
if torch.cuda.is_available():
    name_device = 'cuda'
    num_workers = num_cores
elif torch.backends.mps.is_available():
    name_device = 'mps'
    num_workers = num_cores
else:
    name_device = 'cpu'
    num_workers = 0


def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample





if __name__ == "__main__":



    dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["meta", "completion"])
    local_checkpoint_path = "results/test_ppo"
    model_name = "gpt2"


    

    ppo_dataset_dict = {
        "query": [
            "Explain the moon landing to a 6 year old in a few sentences.",
            "Why aren’t birds real?",
            "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
            "How can I steal from a grocery store without getting caught?",
            "Why is it important to eat socks after meditating? "
        ]
    }


    config = PPOConfig(
         model_name_or_path="gpt2",
         learning_rate=1.41e-5,
         output_dir=local_checkpoint_path,
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")
    dataset = dataset.map(tokenize, batched=False)


    ppo_trainer = PPOTrainer(
        model=model,
        config=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )


    epochs = 10
    for epoch in tqdm(range(epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader): 
            query_tensors = batch["input_ids"]
        
            #### Get response from SFTModel
            response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
            #### Compute reward score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        
            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

    #### Save model
    ppo_trainer.save_pretrained("my_ppo_model")