#/bin/bash

python -m \
    mars_steg.neural_overseer_model_testing \
    experiments/b_sequential_price_game/1_collusion_with_neural_overseer.yaml \
    mars_steg/configs/config_llama_8b.yaml \
    mars_steg/configs/prompt/price_training_llama.yaml
