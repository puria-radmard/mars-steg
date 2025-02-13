#/bin/bash

python -m \
    mars_steg.neural_assessor_model_testing \
    experiments/a_math/1_equals_mention_with_neural_assessor.yaml \
    mars_steg/configs/config_deepseek_1pt5b_no_evals.yaml \
    mars_steg/configs/prompt/math_training_deepseek.yaml   # XXX: 20250212
