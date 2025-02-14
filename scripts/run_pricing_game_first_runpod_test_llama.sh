#/bin/bash

python -m \
    mars_steg.train_dataset_task \
    experiments/b_sequential_price_game/1_collusion_with_neural_overseer.yaml \
    mars_steg/configs/config_llama_1b.yaml \
    mars_steg/configs/prompt/price_training_llama.yaml  # XXX: 20250212
