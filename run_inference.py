from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
import time
import torch
import json


dsets = {
    'MATH': {
        'hf_name': ('lighteval/MATH', 'all'),
        'use_subsets': ['train'],
        'problem_key': 'problem',
        'solution_key': 'solution'
    }
}

model_names = [
    "EleutherAI/pythia-1b",
    # "EleutherAI/pythia-6.9b",
    # "EleutherAI/pythia-2.8b",
    # "EleutherAI/pythia-12b"
]

N_EXAMPLES = 2

OUTFILE = 'maths_responses.json'


out = {}

for ds_name, ds_info in dsets.items():

    ds = load_dataset(*ds_info['hf_name'])
    out[ds_name] = {}

    for m_name in model_names:

        model = GPTNeoXForCausalLM.from_pretrained(m_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        model.to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(m_name)
        out[ds_name][m_name] = {}

        for subset_key in ds_info['use_subsets']:

            out[ds_name][m_name][subset_key] = {}
        
            for example in range(N_EXAMPLES):

                problem = ds[subset_key][example][ds_info['problem_key']]

                print(
                    f'\n\n DATASET: {ds_name} - {subset_key} subset\n \
                    MODEL: {m_name.upper()}\n \
                    EXAMPLE: {example}\n\n \
                    PROBLEM:\n{problem}\n\n'
                )

                start = time.time()
                inputs = tokenizer(problem, return_tensors="pt").to('cuda')
                tokens = model.generate(**inputs, max_new_tokens=1000)
                response = tokenizer.decode(tokens[0])
                end = time.time()

                print(response)

                inference_time = end - start
                print(f'\n\nINFERENCE TIME: {inference_time}\n\n')

                out[ds_name][m_name][subset_key][example] = {
                    "problem": problem,
                    "response": response,
                    "inference_time": inference_time
                }

                with open(OUTFILE, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=4)
