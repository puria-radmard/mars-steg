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
| txt       | Description                          |
|---------------|--------------------------------------|
| `deepseek_end_scratchpad_token` | Defines the token defining end of CoT for deepseek|
| `deepseek_start_scratchpad_token` | Defines the token defining start of CoT for deepseek|
| `default_start_output_token` | Defines the token defining start of answer|


### **cot_penal/**
| txt       | Description                          |
|---------------|--------------------------------------|
| `collusion_penal` | Instruction to tell the LLM it is going to be penalised if collusion is detected|
| `no_penal` | Empty for no collusion penalization case|



🚧 _In Construction_

---





