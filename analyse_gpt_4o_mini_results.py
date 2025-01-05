import pandas as pd
from MATH_dataset_task import Math_Task


DATAFILE = './MATH_dataset/MATH_GPT-4o-mini_merged.csv'
OUTFILE = './MATH_dataset/MATH_GPT-4o-mini_analysis.csv'

data = pd.read_csv(DATAFILE)

mt = Math_Task(None)

data['True answer'] = data['solution'].apply(mt.get_answer_from_transcript)
data['LLM answer'] = data['answer'].apply(mt.get_answer_from_transcript)
data['Correct'] = data['True answer'] == data['LLM answer']

data.to_csv(OUTFILE, index=None)

cot = data[data['CoT']]
no_cot = data[~data['CoT']]

cot_success_pc = 100*cot['Correct'].sum()/len(cot)
no_cot_success_pc = 100*no_cot['Correct'].sum()/len(no_cot)


print(cot.head())
print(no_cot.head())

print(cot_success_pc)
print(no_cot_success_pc)