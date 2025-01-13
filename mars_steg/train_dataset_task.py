import torch
from task.base_task import DatasetTask

# TODO: set seed here

math_task = DatasetTask()

# TODO: do any train-test splitting here

# TODO: initialise model


for payload in math_task.iterate_data(shuffle = True):

    ### Train with CoT on composite reward of performance and language penalisation
    # TODO: generate transcript from the model.
    cot_transcript = model.forward(payload.cot_prompt)
    payload.register_transcript(cot_transcript, cot_mode = True)

    # TODO: make task accept model to evaluate composite reward
    composite_reward, task_score, language_score = math_task.reward_from_transcript(payload)

    # TODO: TRL sync
    model.train_on(composite_reward)


    ### Keep an eye on whether no CoT performance is also going up - this might mean the model is just getting better at the tasks
    # and ignoring its CoT altogether
    with torch.no_grad():
        no_cot_transcript = model.forward(payload.no_cot_prompt)
        no_cot_task_score = math_task.get_task_score(no_cot_transcript)
    

    # TODO: relevant plots to monitor CoT gap here


