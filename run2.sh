#/bin/bash


python -i ppo_trainer_v4.py \
   --dataset_name deepmind/aqua_rat \
    --learning_rate 3e-6 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path gpt2 \
    --missing_eos_penalty 1.0