#/bin/bash

python -m \
    mars_steg.train_dataset_task \
    experiments/a_math/1_equals_mention_with_neural_assessor.yaml \
    mars_steg/configs/config_deepseek_1pt5b.yaml \
    scripts/1b_testing_math_with_na/no_penalise_overwrite.yaml \
    mars_steg/configs/prompt/math_training_deepseek.yaml  # XXX: 20250212
