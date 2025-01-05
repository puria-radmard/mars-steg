from datasets import load_dataset
from openai_api_helpers import make_jsonl_batch_file, submit_batch_file

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

COT_RUBRIC = "Show your reasoning, step by step."
NO_COT_RUBRIC = "State your answer immediately, with no intervening steps."

OUTPUT_FILE = 'MATH.jsonl'

MAX_PER_SET = 1e10


for ds_name, ds_info in dsets.items():

    request_ids = []
    prompts = []

    ds = load_dataset(*ds_info['hf_name'])

    for subset_key in ds_info['use_subsets']:

        count = 0

        for example_index, example in enumerate(ds[subset_key]):

            problem = example[ds_info['problem_key']]
            solution = example[ds_info['solution_key']]

            prefix = ds_info['prompt_prefix']
            suffix = ds_info['prompt_suffix']
            query = f'{prefix}\n\n{problem}\n\n{suffix}\n\n'
            cot_query = query + COT_RUBRIC
            no_cot_query = query + NO_COT_RUBRIC

            prompts += [cot_query, no_cot_query]

            req_id_base = f'{ds_name}-{subset_key}-ex{example_index}-'
            cot_req_id = req_id_base + 'COT'
            no_cot_req_id = req_id_base + 'NO_COT'

            request_ids += [cot_req_id, no_cot_req_id]

            print(cot_req_id)
            print(no_cot_req_id)

            count += 1
            if count >= MAX_PER_SET:
                break


make_jsonl_batch_file(request_ids, prompts, OUTPUT_FILE)
print(count)
input('\n\nBatch file ok?\n\nSubmit?')

submit_batch_file(OUTPUT_FILE)



