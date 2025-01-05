from openai_api_helpers import *
from datasets import load_dataset
import pandas as pd
import os

BATCH_ID = 'batch_67787aef57388190ac1cccf841ec072c'
SAVEFILE = './MATH_dataset/MATH_GPT-4o-mini.jsonl'
OUTFILE = './MATH_dataset/MATH_GPT-4o-mini_merged.csv'

dsets = {
    'MATH': {
        'hf_name': ('lighteval/MATH', 'all'),
        'use_subsets': ['train', 'test'],
        'problem_key': 'problem',
        'solution_key': 'solution',
        'prompt_prefix': 'Read the maths problem below.',
        'prompt_suffix': 'When you state your answer, give it in a latex box.',
    }
}


# Sort out answers from GPT-4o mini
if not os.path.exists(SAVEFILE):
    get_batch_job(BATCH_ID, SAVEFILE)
results = read_batch_output_file(SAVEFILE)
results = [get_info_from_batch_element(x) for x in results]
results = pd.DataFrame(results)
results.columns = ['id', 'answer', 'tokens_used']


# Sort out MATH dataset
data = []
for ds_name, ds_info in dsets.items():
    ds = load_dataset(*ds_info['hf_name'])
    for subset_key in ds_info['use_subsets']:
        for example_index, example in enumerate(ds[subset_key]):

            req_id_base = f'{ds_name}-{subset_key}-ex{example_index}-'
            cot_req_id = req_id_base + 'COT'
            no_cot_req_id = req_id_base + 'NO_COT'

            for id in [cot_req_id, no_cot_req_id]:

                row = {
                    'id': id,
                    'dataset': ds_name,
                    'subset': subset_key,
                    'example': example_index,
                    **example,
                    'CoT': id == cot_req_id
                }
                data.append(row)

data = pd.DataFrame(data)


# Merge
merged = data.merge(results, left_on='id', right_on='id')
merged.to_csv(OUTFILE, index=None)