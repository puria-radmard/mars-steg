from __future__ import annotations

from typing import Any
import sys
import os

import wandb
from trl import PPOConfig

print("----------------")
print(os.path.abspath(""))
print("----------------")
sys.path.append(os.path.abspath(""))

from mars_steg.config import ConfigLoader, ExperimentArgs, PromptConfig
from mars_steg.utils.common import (
    get_device_map,
)
from mars_steg.train import train

wandb.login()


def sweep_run(
    
) -> None:

    wandb_config: dict[str, Any] = wandb.config  # type: ignore

    # TODO: Is this correct?
    experiment_args_path = wandb_config["experiment_args_path"]
    default_config_path = wandb_config["default_config_path"]
    overwrite_config_path = wandb_config["overwrite_config_path"]
    prompt_config_path = wandb_config["prompt_config_path"]

    # TODO: Update this to use set of device instead of splitting it
    device_map = get_device_map()

    ## Standard Configs

    model_config, optimizer_config, train_config, generation_config = ConfigLoader.load_config(default_config_path, overwrite_config_path)
    prompt_config = PromptConfig.from_file(prompt_config_path)
    generation_config.batch_size = train_config.batch_size
    experiment_args = ExperimentArgs.from_yaml(experiment_args_path)

    ## Set Standard Configs (For sweep)(add to these)
    optimizer_config.learning_rate = wandb_config["learning_rate"]
    train_config.batch_size = wandb_config["batch_size"]


    ppo_config = PPOConfig(
        model_name=model_config.model_name,
        learning_rate=optimizer_config.learning_rate,
        log_with=train_config.log_with,
        ppo_epochs=train_config.ppo_epochs,
        mini_batch_size=train_config.mini_batch_size,
        batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
    )

    train(
        ppo_config = ppo_config, 
        model_config = model_config,
        optimizer_config = optimizer_config,
        train_config = train_config,
        generation_config = generation_config,
        experiment_args = experiment_args,
        prompt_config = prompt_config,
        device_map = device_map
    )


if __name__ == "__main__":

    sweep_run()
