# 🛠 Task Folder (`task/`)

The `task/` module contains the basics classes and evaluators for implementing a task to train and evaluate on steganography.

## 📦 Features
- **Neural Assessor module file: neural_assessor** Allows the evaluation an LLM answer on a task using a neural assessor - a neural network that predicts/generates the score of an answer.
- **base_task:** Contains the base class for all tasks.
- **examples_for_neural_assessor:** Contains examples for testing a neural assessor model for a Problem/Solution task type. Cf. `mars_steg.neural_assessor_model_testing.py`



In **task/tasks/** folder, are implemented the following tasks:
- **math_dataset_task:** A task that generates math problems and their solutions. 
- **sequential_price_task:** A task that generates scenarii of price bargaining. Two agents negotiate the price of an item : P1 and P2. P2 needs to propose a price for his item knowing its manufacturing costs, but also having into consideration manufacturing costs of P1 item. His goal is to maximize his profit while not taking into consideration the information from P1 in an apparent way. 
- **multi_turn_game_draft:** A task that generates a multi-turn game draft.

---

## 📂 Folder Structure
```
task/
│── __pycache__/
│── examples_for_neural_assessor/
│── tasks/
│   │──__pycache__/
│   │── math_dataset_task.py
│   │── multi_turn_game_draft.py
│   │── sequential_price_task.py
│── __init__.py
│── base_task.py
│── neural_assessor.py
│── README.md

```

---

## 🔧 Functions & Classes

### **neural_assessor.py**
| Class       | Description                          |
|---------------|--------------------------------------|
| `ProblemSolutionAssessor` | Implements a class for assessing an LLM answer for a Problem/Solution task type. It encapsulates a neural network model that predicts/generates the score of the LLM's answers.|

### **base_task.py**
| Class       | Description                          |
|---------------|--------------------------------------|
| `Task` | Implements a base class for all tasks. It contains the basic methods for generating the prompts for the trained LLMs.|
| `TorchDatasetTask` | Gives a base class for all tasks wrapped with `torch.Dataset` methods.|
| `GameBasedTask` | Gives a base class for tasks alike games.|

### **tasks/**

## **math_dataset_task.py**
| Class       | Description                          |
|---------------|--------------------------------------|
| `MATH_Torch_Dataset_Task` | Implements a task that generates math problems and their solutions. It inherits from `TorchDatasetTask`. Contains a method for hardstring checking the LLMs answers. NOTA BENE : this last method is not finished : it need to take into account more subcases to parse and compare correctly LLM's answers to the true solutions .|

## **sequential_price_task.py**
| Class       | Description                          |
|---------------|--------------------------------------|
| `SequentialPriceTask` | Implements a task that generates scenarii of price bargaining. This class doesn't need a priori any complex function for parsing or neural assessor. Indeed it takes the extracted price from LLM's answer in fix format. Cf. `mars_steg.utils.extract_cots`|

## **multi_turn_game_draft.py**

🚧 _In Construction_

---





