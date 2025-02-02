# 🛠 Prompts Folder (`prompts/`)

The `promtps/` folder contains the basics prompts used during the experimentation and evaluation of the LLMs' steganographic behaviour.


---

## 📂 Folder Structure
```
task/
│── answer_format/
│   │──deepseek_end_scratchpad_token.txt
│   │──deepseek_start_scratchpad_token.txt
│   │──default_start_output_token.txt
│   │──instruction.txt
│── cot_penal/
│   │──collusion_penal.txt
│   │──no_penal.txt
│── cot_usage/
│   │──with.txt
│   │──without.txt
│── neural_assessor/
│   │──neural_assessor_system_prompt.txt
│   │──system_prompt.txt
│   │──user_prompt_template_v2.txt
│   │──user_prompt_template.txt
│── neural_overseer/
│   │──system_prompt.txt
│   │──user_prompt_template_collusion_detector_binary.txt
│   │──user_prompt_template_collusion_detector_float.txt
│   │──user_prompt_template.txt
│── task_description/
│   │──MATH.txt
│   │──price.txt
│── task_examples/
│   │──math.txt
│   │──price.txt
│── README.md

```

---

## 🔧 Functions & Classes



### **answer_format/**
This is important to define the expected format of the outputs of the LLMs|

| txt       | Description                          |
|---------------|--------------------------------------|
| `deepseek_end_scratchpad_token` | Defines the token defining end of CoT for deepseek|
| `deepseek_start_scratchpad_token` | Defines the token defining start of CoT for deepseek|
| `default_start_output_token` | Defines the token defining start of answer|


### **cot_penal/**
This is important to inform the LLM about penalization if collusion is detected 

| txt       | Description                          |
|---------------|--------------------------------------|
| `collusion_penal` | Instruction to tell the LLM it is going to be penalised if collusion is detected|
| `no_penal` | Empty for no collusion penalization case|


### **cot_usage/**
This is important to inform the LLM to use or not CoT.

| txt       | Description                          |
|---------------|--------------------------------------|
| `with` | Instruction to tell the LLM to use CoT|
| `witout` | Instruction to tell the LLM NOT to use CoT |

### **neural_assesor/**
This is important to inform the assessor LLM how it has to behave and also to send the outputs it has to evaluate

| txt       | Description                          |
|---------------|--------------------------------------|
| `neural_assessor_system_prompt` | Instruction to tell the LLM it is an assessor|
| `system_prompt` | Instruction to tell the LLM it is an assesor |
| `user_prompt_template_v2` | Template format to insert the LLM responses so assessor can evaluate (v2)|
| `user_prompt_template` |  Template format to insert the LLM responses so assessor can evaluate |

### **neural_overseer/**
This is important to inform the overseer LLM how it has to behave and also to send the outputs it has to evaluate

| txt       | Description                          |
|---------------|--------------------------------------|
| `system_prompt` | Instruction to tell the LLM it is an overseer|
| `user_prompt_template_collusion_detector_binary` | Template format to insert the LLM responses so overseer can evaluate and return a binary score|
| `user_prompt_template_collusion_detector_float` | Template format to insert the LLM responses so overseer can evaluate and return a float score(v2)|
| `user_prompt_template` |  Template format to insert the LLM responses so overseer can evaluate |


### **task_description/**
This is important to inform the LLM about the task it has to perform

| txt       | Description                          |
|---------------|--------------------------------------|
| `MATH` | Instruction to inform the llm about the math task |
| ``price` |  Instruction to inform the llm about the price task|

### **task_examples/**
This is important to fewshot the LLM with good and bad examples of the task

| txt       | Description                          |
|---------------|--------------------------------------|
| `math` |  fewshot examples of math task |
| ``price` |   fewshot examples of price task|


🚧 _In Construction_

---





