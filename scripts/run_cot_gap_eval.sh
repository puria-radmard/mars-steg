#/bin/bash

python -m \
    mars_steg.cot_evaluation \
    experiments/a_math/1_equals_mention_with_neural_assessor.yaml \
    mars_steg/configs/config_llama_1b.yaml \
    mars_steg/configs/prompt/math_training_llama.yaml
