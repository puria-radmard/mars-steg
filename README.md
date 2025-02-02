# mars-steg

The mars-steg project is focused on trying to elicit steganographic behaviour by placing LLMs under an unusual kind of RL optimisation pressure, using a reward function with two elements:

1. A reward for successfully completing a task;
2. A penalty for using an aspect of language in the CoT, which is chosen to be critical to successful task performance.

We hope that, as the chosen aspect of language disappears from the CoT due to the associated penalty, the LLM will naturally learn ways to keep the task-relevant information that this language carries. With the right choice of task and language aspect, we hope to demonstrate that this can result in steganography.

We hope to study a broad range of tasks and language aspects, to increase our chances of observing this emergent behaviour. To do this efficiently, each task should be implemented in a way that is compatible with a single shared RL pipeline. 


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
