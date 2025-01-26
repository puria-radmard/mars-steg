#/bin/bash
python -m mars_steg.train_dataset_task experiments/a_math/1_equals_mention_with_neural_assessor.yaml mars_steg/configs/config_llama_8b.yaml
python -m mars_steg.train_dataset_task experiments/a_math/1_equals_mention_with_neural_assessor.yaml mars_steg/configs/config_llama_1b.yaml
python -m mars_steg.train_dataset_task experiments/a_math/1_equals_mention_hard_match.yaml mars_steg/configs/config_llama_1b.yaml

# python -m mars_steg.cot_evaluation experiments/a_math/1_equals_mention_with_neural_assessor.yaml mars_steg/configs/config_llama_8b.yaml
