import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
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

if __name__ == "__main__":

    print("--------------------")
    print(f"Device: {name_device}")
    print(f"Number of workers: {num_workers}")
    print("--------------------")
    
    dataset_name = "deepmind/aqua_rat"
    logging_dir = os.path.join("results/", "logs")
    dataset = load_dataset(dataset_name, "raw")
    


    local_checkpoint_path = "results/test_ppo"
    # config =  PPOConfig(
    #         output_dir=local_checkpoint_path,
    #         num_train_epochs=10,
    #         per_device_train_batch_size=4,
    #         per_device_eval_batch_size=1,
    #         save_steps=500,
    #         save_total_limit=3,
    #         learning_rate=3e-4,
    #         logging_steps=1000,
    #         warmup_steps=500,
    #         weight_decay=0.01,
    #         fp16=True,  # Mixed precision
    #         gradient_accumulation_steps=4,  
    #         report_to="none",
    #         optim="adamw_torch",
    #         logging_dir=os.path.join(local_checkpoint_path, "logs"),
    #     )

    TRAINING_SIZE = 100
    SEED = 42 # Deterministic execution 

    # Tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token if not present



    # 2. Load the GPT-2 model and wrap it for PPO
    model = GPT2LMHeadModel.from_pretrained(model_name).to(name_device) 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = PPOConfig(
        learning_rate=1.41e-5,
        output_dir=local_checkpoint_path,
    )

    #Reward model
    reward_model = GPT2LMHeadModel.from_pretrained(config.reward_model_path).to(name_device)

    # 3. Load the training dataset
    train_dataset = dataset["train"].select(range(TRAINING_SIZE))
    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=num_workers
    )


    def compute_reward(inputs, outputs):
        # Implement custom logic to compute reward based on inputs and outputs
        # For example, comparing outputs to high-quality references or using a classifier
        return torch.tensor([1.0] * len(inputs)).to(outputs.device)  # Placeholder reward

    # 5. Initialize PPO trainer
    ref_model = GPT2LMHeadModel.from_pretrained(model_name).to(name_device) 
    ppo_trainer = PPOTrainer(
        model=model,  
        ref_model=ref_model,
        reward_model=reward_model,
        config=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,

    )
    # 6. Training loop
    num_epochs = 3

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch in data_loader:
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()

            # Generate responses
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=128,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Compute rewards
            rewards = compute_reward(input_ids, outputs)

            # Run PPO step
            stats = ppo_trainer.step(input_ids, outputs, rewards)

            # Log stats
            print(stats)

        # Save the model checkpoint after each epoch
        model.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"ppo_gpt2_epoch_{epoch + 1}")

    print("Training complete!")
