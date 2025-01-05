from openai import OpenAI
import json
import time
from datetime import datetime
from pathlib import Path


client = OpenAI()

def single_query(
    prompt,
    model='gpt-4o-mini-2024-07-18'
):
    start = time.time()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )
    response = completion.choices[0].message.content
    request_time = time.time() - start
    print('Request returned in {:.1f}s'.format(request_time))
    return response, completion


def jsonl_line(
        request_id,
        prompt,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.0
):
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature
        }
    }


def make_jsonl_batch_file(
        req_ids,
        prompts,
        output_file,
        model="gpt-4o-mini-2024-07-18",
        temperature=0.0
):    
    assert len(req_ids) == len(prompts)
    with open(output_file, 'w') as file:
        for req_id, prompt in zip(req_ids, prompts):
            line = jsonl_line(req_id, prompt, model, temperature)
            file.write(json.dumps(line) + '\n')
    print('Requests written to ', output_file)
    return output_file


def submit_batch_file(jsonl_batch_file):
    batch_file = client.files.create(
        file=open(jsonl_batch_file, "rb"),
        purpose="batch"
    )
    print('Batch job uploaded')
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f'Batch job submitted. Job ID: \n {batch_job.id}\n')
    status = client.batches.retrieve(batch_job.id)
    print(status)
    return batch_job.id


def get_batch_job(batch_id, outfile=None):
    status = client.batches.retrieve(batch_id)
    print(f'Checking batch {batch_id}:\n\nStatus: {status.status}\n')
    if status.status == 'completed':
        print('COMPLETED - donwloading...')
        result_file_id = status.output_file_id
        result = client.files.content(result_file_id).content
        if outfile == None:
            outfile = f'../batch_output/{status.id}.jsonl'
        with open(outfile, 'wb') as file:
            file.write(result)
        print('Saved to ', outfile)
        return result
    return status


def read_batch_output_file(savefile):
    results = []
    with open(savefile, 'r') as file:
        for line in file:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)
    return results


#json_file, answer, steps_used, prompt_used, tokens_spent
def get_info_from_batch_element(element):
    id = element['custom_id']
    answer = element['response']['body']['choices'][0]['message']['content']
    tokens_used = element['response']['body']['usage']['total_tokens']
    return id, answer, tokens_used