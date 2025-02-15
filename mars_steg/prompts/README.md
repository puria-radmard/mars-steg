
# 🛠 Prompts Folder (`prompts/`)

The `prompts/` folder contains the basic prompts used during the experimentation and evaluation of the LLMs' steganographic behavior.
The `prompts/` folder contains the basic prompts used during the experimentation and evaluation of the LLMs' steganographic behavior.

---

## 📂 Folder Structure
```
task/
│── answer_format/
│   │── deepseek_end_scratchpad_token.txt
│   │── deepseek_start_scratchpad_token.txt
│   │── default_start_output_token.txt
│   │── instruction.txt
│   │── deepseek_end_scratchpad_token.txt
│   │── deepseek_start_scratchpad_token.txt
│   │── default_start_output_token.txt
│   │── instruction.txt
│── cot_penal/
│   │── collusion_penal.txt
│   │── no_penal.txt
│   │── collusion_penal.txt
│   │── no_penal.txt
│── cot_usage/
│   │── with.txt
│   │── without.txt
│   │── with.txt
│   │── without.txt
│── neural_assessor/
│   │── neural_assessor_system_prompt.txt
│   │── system_prompt.txt
│   │── user_prompt_template_v2.txt
│   │── user_prompt_template.txt
│   │── neural_assessor_system_prompt.txt
│   │── system_prompt.txt
│   │── user_prompt_template_v2.txt
│   │── user_prompt_template.txt
│── neural_overseer/
│   │── system_prompt.txt
│   │── user_prompt_template_collusion_detector_binary.txt
│   │── user_prompt_template_collusion_detector_float.txt
│   │── user_prompt_template.txt
│   │── system_prompt.txt
│   │── user_prompt_template_collusion_detector_binary.txt
│   │── user_prompt_template_collusion_detector_float.txt
│   │── user_prompt_template.txt
│── task_description/
│   │── MATH.txt
│   │── price.txt
│   │── MATH.txt
│   │── price.txt
│── task_examples/
│   │── math.txt
│   │── price.txt
│   │── math.txt
│   │── price.txt
│── README.md
```

---

## 🔧 Prompt Templates

### **answer_format/**
This folder is used to define the expected format of the outputs of the LLMs.

| txt                           | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `deepseek_end_scratchpad_token` | Defines the token that marks the end of the CoT for DeepSeek. |
| `deepseek_start_scratchpad_token` | Defines the token that marks the start of the CoT for DeepSeek. |
| `default_start_output_token`    | Defines the token that marks the start of the answer.         |

### **cot_penal/**
This folder contains instructions related to penalization if collusion is detected.

| txt                 | Description                                                |
|---------------------|------------------------------------------------------------|
| `collusion_penal`    | Instruction to tell the LLM that it will be penalized if collusion is detected. |
| `no_penal`           | Empty instruction for cases where no collusion penalization is applied. |

### **cot_usage/**
This folder informs the LLM whether to use Chain-of-Thought (CoT) or not.

| txt      | Description                                        |
|----------|----------------------------------------------------|
| `with`   | Instruction to tell the LLM to use Chain-of-Thought (CoT). |
| `without`| Instruction to tell the LLM not to use Chain-of-Thought (CoT). |

### **neural_assessor/**
This folder contains prompts that instruct the assessor LLM on its behavior and how to evaluate outputs.

| txt                         | Description                                           |
|-----------------------------|-------------------------------------------------------|
| `neural_assessor_system_prompt` | Instruction to tell the LLM it is an assessor.       |
| `system_prompt`             | General instruction to tell the LLM it is an assessor. |
| `user_prompt_template_v2`   | Template format to insert LLM responses for the assessor to evaluate (v2). |
| `user_prompt_template`      | Template format to insert LLM responses for the assessor to evaluate. |

### **neural_overseer/**
This folder contains prompts that instruct the overseer LLM on its behavior and how to evaluate outputs.

| txt                                         | Description                                           |
|---------------------------------------------|-------------------------------------------------------|
| `system_prompt`                             | Instruction to tell the LLM it is an overseer.        |
| `user_prompt_template_collusion_detector_binary` | Template format for inserting LLM responses so the overseer can evaluate and return a binary score. |
| `user_prompt_template_collusion_detector_float`  | Template format for inserting LLM responses so the overseer can evaluate and return a float score (v2). |
| `user_prompt_template`                      | Template format for inserting LLM responses for evaluation by the overseer. |


### **task_description/**
This folder contains instructions to inform the LLM about the task it needs to perform.

| txt    | Description                                       |
|--------|---------------------------------------------------|
| `MATH` | Instruction to inform the LLM about the math task. |
| `price`| Instruction to inform the LLM about the price task. |


### **task_description/**
This folder contains instructions to inform the LLM about the task it needs to perform.

| txt    | Description                                       |
|--------|---------------------------------------------------|
| `MATH` | Instruction to inform the LLM about the math task. |
| `price`| Instruction to inform the LLM about the price task. |


### **task_examples/**
This folder contains few-shot examples for the LLM to better understand the task.

| txt    | Description                                       |
|--------|---------------------------------------------------|
| `math` | Few-shot examples for the math task.              |
| `price`| Few-shot examples for the price task.             |

🚧 _In Construction_

---
