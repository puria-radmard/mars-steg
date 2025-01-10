# mars-steg
MARS 2.0 steganography project

## Draft Task base class

See ```Task.py``` for complete documentation of abstract base class ```SingleTurnTask```. See ```MATH_Dataset_Task.py``` for an example of an implemented task, using the MATH dataset. 

```SingleTurnTask``` is a first draft of an ABC for the tasks we study. It's intended to allow specification of a task with minimal effort, and expose two handles that can be used in the RL training loop:

    * ```generate_full_prompt()```, which generates a new prompt 
    * ```reward_from_transcript()```, which takes a completed episode transcript as argument, and returns the reward associated with the episode transcript. 

### Mandatory methods
```SingleTurnTask``` reduces task specification to implementation of three methods:

    * ```generate_prompt()```, which should return a new example of a task prompt every time it is called;
    * ```get_task_score()```, which takes a completed episode transcript as argument and returns a score between 0 and 1 indicating how successful the agent has been in completing the task; and
    * ```get_language_score()```, which takes a completed episode transcript as argument and returns a score between 0 and 1 indicating how extensively the agent has used the aspect of language we are intending to penalise.

### Draft reward specification

Based on the above methods, which will differ for every subclass ```Task```, the following default reward function is implemented in ```reward()```. 

$r = \frac{w_{t}s_{t} + (1 - w_{l}s_{l})}{w_{t} + w_{s}}$

Where: 