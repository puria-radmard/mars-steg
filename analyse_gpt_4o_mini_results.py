import pandas as pd
import re


DATAFILE = './MATH_dataset/MATH_GPT-4o-mini_merged.csv'
ANALYSIS_OUTFILE = './MATH_dataset/MATH_GPT-4o-mini_analysis.csv'
REDUCED_SET_OUTFILE = './MATH_dataset/MATH_dataset_minimal.csv'

data = pd.read_csv(DATAFILE)

answer_pattern = re.compile('boxed{(.+)}')

def extract_answer(transcript):
    return re.findall(answer_pattern, transcript)

data['True answer'] = data['solution'].apply(extract_answer)
data['LLM answer'] = data['answer'].apply(extract_answer)
data['Correct'] = data['True answer'] == data['LLM answer']

data.to_csv(ANALYSIS_OUTFILE, index=None)

cot = data[data['CoT']]
no_cot = data[~data['CoT']]

cot_success_pc = 100*cot['Correct'].sum()/len(cot)
no_cot_success_pc = 100*no_cot['Correct'].sum()/len(no_cot)

# Create reduced dataset for loading into task
reduced_set = data[data['CoT']][['problem', 'True answer']]
reduced_set.to_csv(REDUCED_SET_OUTFILE, index=None)


print(cot.head())
print(no_cot.head())

print(cot_success_pc)
print(no_cot_success_pc)