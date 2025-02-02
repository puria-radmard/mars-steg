# 🛠 Mars_steg folder (`mars-steg/mars_steg`)

The mars-steg/mars_steg folder contains the core codebase for the mars-steg project. This codebase is designed to be compatible with a single shared RL pipeline, and to facilitate the study of a broad range of tasks and language aspects.


## 📦 Features

- **Task Classes**: Base classes for defining new tasks, and for defining new aspects of language to penalise.
- **Language Aspect**: Base classes for defining aspects of language to penalise.
- **Prompt Data**: Dataclasses for storing information about prompts and responses.
- **configs**: Configuration files for defining task and language aspect combinations.
- **utils**: General helper functions and decorators.
- **models**: Classes for defining models, wrapping Hugginface models, for the RL pipeline, the overseer and the neural assessor.
-**prompts**: Folder containing prompts formats and information for each task/model.


---

## 📂 Folder Structure
```
mars_steg/
│── configs/          # Configuration files for defining task and language aspect combinations (yaml)
│── language/         # Base classes for defining aspects of language to penalise
│── datasets/         # Folder containing datasets for each task. For SequencePricing game, a generator script is provided.
│── models/           # Classes for defining models, wrapping Hugginface models, for the RL pipeline, the overseer and the neural assessor.
│── prompts/          # Folder containing prompts formats and information for each task/model (.txt files).
│── task/            # Base classes for defining task. Contains also task class for Math Problem solving and Sequenc ePricing game.
│── utils/            # General helper functions and decorators.
│── benchmark.py      🚧 _In Construction_ 
│── config.py         # Python file containing config dataclasses
│── cot_evaluation.py  # Script for evaluating the CoT gap
│── neural_assessor_model_testing.py # Script for testing on a HF model in the role of the neural assessor
│── overseer_model_testing.py 🚧 _In Construction_ 
│── train_dataset_task.py # Script for launching an experiment on a task. Contains the RL pipeline.


```

---

## 📖 Related Documentation
- 📜 [Main Documentation](../../docs/index.md)
- 🏗️ [Sphinx API Docs](../../docs/source/api/utils.rst)

---

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

---

### To Do

#### Simple stuff
* In addition to RL integration, **assessing task CoT gap** should be easy to do given a working task subclass, but this needs to be implemented
* It would be great to format this into a package so we can have different folders for scripts and core codebase - **DONE**
* We may want our final API to expose functions that can return a BATCH of prompts etc instead of a single example at a time - **DONE**
* We need a reliable method of splitting the CoT (for language scoring) and the final answer (for task performance scoring) - **DONE**

#### Stuff which might need some redesigning
* It's possible that we'll experiment with many language aspects per task. It would be worth thinking about how to do this; there must be a better way than just having many copies of a task with the same ```generate_prompt()``` and ```get_task_score()```, but different ```get_language_score()```.
    * Simple ordinal system with if statements? - **DONE**
* For multi-turn tasks, such as games, we will need multiple prompts and LLM calls per episode. At the moment, ```Task``` cannot support this. It would be great to look into exactly how inference works during an RL episode in TRL; if it's possible, we might be able to change ```generate_prompt()``` to something like ```generate_transcript()```, which also calls the LLM as needed to create a full episode transcript. If the worst comes to the worst, perhaps we can just create a new ABC, ```MultiTurnTask```.