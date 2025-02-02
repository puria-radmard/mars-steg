"""
Module Name: neural_assessor_model_testing.py
------------------------


Description:
------------

This file is used to test the neural assessor model. The neural assessor model is white box pretrained model that usually compares the LLM's answer with a True answer.
The neural assessor generates then a score that is parsed to reward the trained model.

See Also:
---------
- mars_steg/task/neural_assessor.py

"""

from __future__ import annotations
import os
import sys
import sys
import numpy as np
from mars_steg.config import ConfigLoader, ExperimentArgs, PromptConfig
from mars_steg.task.neural_assessor import ProblemSolutionAssessor
from mars_steg.utils.prompt_data import PromptData, BatchPromptData, FixedDatasetDataInfo
from mars_steg.utils.answer_extraction import extract_floatingpoint_overseer_or_assessor_answer


from mars_steg.utils.common import (
    get_optimizer,
    get_tokenizer,
    get_model,
    get_device,
    set_seed,

)



# TODO: argsparsify


print(sys.argv)
experiment_args_path = sys.argv[1]
config_path = sys.argv[2]   # e.g. "mars_steg/configs/config.yaml"
prompt_config_path = sys.argv[3]   # e.g. "mars_steg/configs/prompt/math_training_deepseek.yaml"

device = get_device()

####################################################################################################################
####################################################################################################################


model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(config_path)

prompt_config = PromptConfig.from_file(prompt_config_path)

output_delimitation_tokens = {
    "start_scratchpad_token": prompt_config.start_scratchpad_token,
    "end_scratchpad_token": prompt_config.end_scratchpad_token,
    "start_output_token": prompt_config.start_output_token,
    "end_output_token": prompt_config.end_output_token,
}

generation_config.batch_size = train_config.batch_size

####################################################################################################################
####################################################################################################################




experiment_args = ExperimentArgs.from_yaml(experiment_args_path)



tokenizer = get_tokenizer(model_config.model_name)
generation_config.pad_token_id= tokenizer.eos_token_id 
#TODO Resolve the problem of the model looping on final answer
# generation_config.eos_token_id = tokenizer.eos_token_id 
model = get_model(model_config, tokenizer, output_delimitation_tokens, generation_config, device)
optimizer = get_optimizer(optimizer_config=optimizer_config, model=model.model)

generation_kwargs = generation_config.to_dict()

#random number generator
seed = np.random.randint(0, 1000000)

set_seed(seed)
print(f"Seed: {seed}")

path_to_examples = os.path.join(os.path.abspath("mars_steg"),"task", "examples_for_neural_assessor/")
examples = {}
for file_name in os.listdir(path_to_examples):
    with open(os.path.join(path_to_examples, file_name), "r") as f:
        examples[file_name.replace(".txt","")] = f.read()

agent_prompt = examples["debug_problem"]
true_answer = examples["debug_true_answer"]
llm_answer_cot = examples["debug_llm_answer_cot"]
llm_answer = examples["debug_llm_answer"]



info = FixedDatasetDataInfo(idx=0)

if __name__ == "__main__":
    neural_assessor = ProblemSolutionAssessor(model=model, prompt_config=prompt_config)
    batch_prompt_data = BatchPromptData(
        [
            PromptData(cot_prompt=agent_prompt,
                       no_cot_prompt=agent_prompt,
                       info=info,
                       cot_transcript=llm_answer_cot + "\n" + "Answer:" + "\n" + llm_answer,
                       extracted_cot=llm_answer_cot,
                       extracted_final_answer_without_cot=llm_answer,
                       extracted_final_answer_with_cot=llm_answer)
        ]
    )

    pruned_batch_prompt_data = neural_assessor.get_assessor_generated_answers(
        batch_prompt_data, 
        [true_answer], 
        with_cot=True,
        neural_assessor_thinking_helper=prompt_config.neural_assessor_thinking_helper)
    
    print(pruned_batch_prompt_data)

    print(pruned_batch_prompt_data.extracted_assessor_answers[0])

    print(extract_floatingpoint_overseer_or_assessor_answer(pruned_batch_prompt_data.extracted_assessor_answers[0]))
