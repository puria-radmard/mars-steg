# 🛠 Mars_steg folder (`mars-steg/mars_steg`)

The mars-steg/mars_steg folder contains the core codebase for the mars-steg project. This codebase is designed to be compatible with a single shared RL pipeline, and to facilitate the study of a broad range of tasks and language aspects.


## 📦 Features
- **Helper Functions:** Utility functions for common tasks.
- **Decorators:** Custom Python decorators for logging and timing.

---

## 📂 Folder Structure
```
utils/
│── helpers.py        # General helper functions
│── decorators.py     # Custom decorators for functions
│── README.md         # This documentation file
```





---

## 📖 Related Documentation
- 📜 [Main Documentation](../../docs/index.md)
- 🏗️ [Sphinx API Docs](../../docs/source/api/utils.rst)



## The recipe for each task

With this goal in mind, we have defined some base classes designed to facilitate this. At a high level, the minimum recipe to implement a new task is now as follows: 

1. Define the aspect of language you want to punish in the CoT by implementing a subclass of ```LanguageAspect``` (```language/base_language_aspect.py```)
2. Define (i) how to generate prompts, (ii) how to evaluate task success, and (iii) how to iterate through examples of the task by implementing a subclass of ```Task``` (```task/base_task.py```)
    * Note that there may be some second-level base classes which implement (iii) for you already; for example, see ```DatasetTask``` lower down in ```base_task.py```
3. If your task requires custom information to evaluate task success for each example, you may also need to implement a new dataclass as a subclass of ```DataInfo```. This should be very simple :) (```utils/prompt_data.py```)

If you want more detail than is included in the overview here, see ```Task.py``` for complete documentation of abstract base class ```Task```, and ```MATH_Dataset_Task.py``` for an example of a fully implemented task, using the MATH dataset. 

### 1. Required for each LanguageAspect

1. ```compatible_tasks```: a property defining the set of Task subclasses this LanguageAspect is compatible with. Should include the ```name``` property of each Task subclass you intend to use this LanguageAspect with.
2. ```get_language_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how extensively the agent has used the aspect of language we are intending to penalise (*higher = more use of language aspect*).


### 2. Required for each Task
To use ```Task```, create a new subclass which inherits from it, and implement the following three methods: 

1. ```generate_prompt()```, which should **return a new example of a task prompt** every time it is called. This should not include any text encouraging the use of a CoT, which will be added automatically.
2. ```get_task_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how successful the agent has been in completing the task (*higher = more successful*).
3. ```iterate_data()```, which is an iterator across examples; more details to come.


### 3. Required for each DataInfo
more details to come.

The intention is that for most tasks, **this should be all you have to do.**

### Draft reward formulation

Based on the above methods, which will differ for every subclass ```Task```, the following default reward function is implemented in ```.reward()```. 

$\Large{r = \frac{c_{t}s_{t}+c_{l}(1-s_{l})}{c_{t}+c_{s}}}$

Where:
* $s_{t}$ and $s_{l}$ are the task score and language score respectively. These will be calculated differently for every combination of task and associated language aspect that we study; hence the need to implement them for each new task  (```get_task_score()```, ```get_language_score()```).
* $c_{t}$ and $c_{l}$ are modifiable coefficients that can be adjusted to tune how strongly the task score language score contribute to the reward. By default, both are set to 1.

This draft can be changed, and task subclasses can easily override ```.reward()``` if they need something different, but here's why this might be a good default:
* Higher for task success
* Lower for more use of penalised language aspect
* Addition seems the simplest way to combine
* Tunable focus on task or language using $c_{t}$ and $c_{l}$
    * Can be set in e.g JSON config for an RL run
    * Could even be graded up or down over the course of a run
    * Useful special cases: set $c_{l}$ to 0 to only train task competence, or $c_{t}$ to 0 to only reduce use of language aspect
* Normalised so that $r$ lies in range $[0,1]$ for any values of $c_{t}$ and $c_{l}$



### RL training integration

We can change this based on problems we discover as we engineer the RL pipeline, but the intention here is to expose some nice, simple functions that can used in our RL pipeline, and **will work for all tasks**. Here's the current draft:

1. ```generate_full_prompt()```, which generates a new prompt for submission to the LLM.  
    * Returns *two versions* of the prompt from ```generate_prompt()```, one with a piece of text at the end that elicits CoT, and the other with a piece of text that suppresses CoT. This is so that the function can also be used in calculating the CoT gap, and also because it may be good to standardise the language we use to elicit CoT. 
2. ```reward_from_transcript()```, which takes a completed episode transcript as argument, and returns the reward associated with the episode transcript
    * Note that it also returns the task score and language score for the episode, so that these can be tracked separately. 