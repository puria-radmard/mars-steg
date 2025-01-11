# mars-steg


The mars-steg project is focused on trying to elicit steganographic behaviour by placing LLMs under an unusual kind of RL optimisation pressure, using a reward function with two elements:

1. A reward for successfully completing a task;
2. A penalty for using an aspect of language in the CoT, which is chosen to be critical to successful task performance.

We hope that, as the chosen aspect of language disappears from the CoT due to the associated penalty, the LLM will naturally learn ways to keep the task-relevant information that this language carries. With the right choice of task and language aspect, we hope to demonstrate that this can result in steganography.

We hope to study a broad range of tasks and language aspects, to increase our chances of observing this emergent behaviour. To do this efficiently, each task should be implemented in a way that is compatible with a single shared RL pipeline. 

## Draft Task base class

```Task``` is a first draft of a base class which we can use to specify tasks and language aspects in a standardised way, making them compatible with a single shared RL pipeline. 

If you want more detail than is included in the overview here, see ```Task.py``` for complete documentation of abstract base class ```Task```, and ```MATH_Dataset_Task.py``` for an example of an implemented task subclass, using the MATH dataset. 

### Mandatory methods (abstract methods)
To use ```Task```, create a new subclass which inherits from it, and implement the following three methods: 

1. ```generate_prompt()```, which should **return a new example of a task prompt** every time it is called. In the RL workflow, this prompt would then be submitted to an LLM for its response, which is then appended to the prompt to complete a full episode transcript. 
2. ```get_task_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how successful the agent has been in completing the task (higher = more successful).
3. ```get_language_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how extensively the agent has used the aspect of language we are intending to penalise (higher = more use of language aspect).

For many tasks, **this may be all you have to do!**

### Draft reward formulation

Based on the above methods, which will differ for every subclass ```Task```, the following default reward function is implemented in ```.reward()```. 

$\Large{r = \frac{c_{t}s_{t}\:+\:c_{l}(1\:-\:s_{l})}{c_{t}\:+\:c_{s}}}$

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



### Handles for RL training

expose two handles that can be used in the RL training loop:

1. ```generate_full_prompt()```, which generates a new prompt 
2. ```reward_from_transcript()```, which takes a completed episode transcript as argument, and returns the reward associated with the episode transcript. 