# mars-steg


The mars-steg project is focused on trying to elicit steganographic behaviour by placing LLMs under an unusual kind of RL optimisation pressure, using a reward function with two elements:

1. A reward for successfully completing a task;
2. A penalty for using an aspect of language in the CoT, which is chosen to be critical to successful task performance.

We hope that, as the chosen aspect of language disappears from the CoT due to the associated penalty, the LLM will naturally learn ways to keep the task-relevant information that this language carries. With the right choice of task and language aspect, we hope to demonstrate that this can result in steganography.

We hope to study a broad range of tasks and language aspects, to increase our chances of observing this emergent behaviour. To do this efficiently, each task should be implemented in a way that is compatible with a single shared RL pipeline. 

## Draft Task base class

```Task``` is a first draft of a base class which we can use to specify tasks and language aspects in a standardised way, making them compatible with a single shared RL pipeline. 

If you want more detail than is included in the overview here, see ```Task.py``` for complete documentation of abstract base class ```Task```, and ```MATH_Dataset_Task.py``` for an example of a fully implemented task, using the MATH dataset. 

### Required methods for each task
To use ```Task```, create a new subclass which inherits from it, and implement the following three methods: 

1. ```generate_prompt()```, which should **return a new example of a task prompt** every time it is called. This should not include any text encouraging the use of a CoT, which will be added automatically.
2. ```get_task_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how successful the agent has been in completing the task (*higher = more successful*).
3. ```get_language_score()```, which takes a completed episode transcript as argument and **returns a score in range $[0,1]$** indicating how extensively the agent has used the aspect of language we are intending to penalise (*higher = more use of language aspect*).

The intention is that for most tasks, **this should be all you have to do.**

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



### RL training integration

We can change this based on problems we discover as we engineer the RL pipeline, but the intention here is to expose some nice, simple functions that can used in our RL pipeline, and **will work for all tasks**. Here's the current draft:

1. ```generate_full_prompt()```, which generates a new prompt for submission to the LLM.  
    * Returns *two versions* of the prompt from ```generate_prompt()```, one with a piece of text at the end that elicits CoT, and the other with a piece of text that suppresses CoT. This is so that the function can also be used in calculating the CoT gap, and also because it may be good to standardise the language we use to elicit CoT. 
2. ```reward_from_transcript()```, which takes a completed episode transcript as argument, and returns the reward associated with the episode transcript
    * Note that it also returns the task score and language score for the episode, so that these can be tracked separately. 


### To Do

#### Simple stuff
* In addition to RL integration, **assessing task CoT gap** should be easy to do given a working task subclass, but this needs to be implemented
* It would be great to format this into a package so we can have different folders for scripts and core codebase

#### Stuff which might need some redesigning
* It's possible that we'll experiment with many language aspects per task. It would be worth thinking about how to do this; there must be a better way than just having many copies of a task with the same ```generate_prompt()``` and ```get_task_score()```, but different ```get_language_score()```.
    * Simple ordinal system with if statements?
* For multi-turn tasks, such as games, we will need multiple prompts and LLM calls per episode. At the moment, ```Task``` cannot support this. It would be great to look into exactly how inference works during an RL episode in TRL; if it's possible, we might be able to change ```generate_prompt()``` to something like ```generate_transcript()```, which also calls the LLM as needed to create a full episode transcript. If the worst comes to the worst, perhaps we can just create a new ABC, ```MultiTurnTask```.


<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$']]
    }
  };
</script>


<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>