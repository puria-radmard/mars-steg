# Train on Multiple GPUs / Nodes

The trainers in TRL use 🤗 Accelerate to enable distributed training across multiple GPUs or nodes. To do so, first create an 🤗 Accelerate config file by running:

```bash
accelerate config
```

and answering the questions according to your multi-GPU / multi-node setup. You can then launch distributed training by running:


```bash
accelerate launch your_script.py
```

To use the template configs in this dir, simply pass the path to the config file when launching a job, e.g.:
```bash
accelerate launch --config_file=examples/accelerate_configs/multi_gpu.yaml --num_processes {NUM_GPUS} path_to_script.py --all_arguments_of_the_script
```

# Distributed Training with DeepSpeed

All of the trainers in TRL can be run on multiple GPUs together with DeepSpeed ZeRO-{1,2,3} for efficient sharding of the optimizer states, gradients, and model weights. To do so, run:

```bash
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_your_script.py --all_arguments_of_the_script
```

Note that for ZeRO-3, a small tweak is needed to initialize your reward model on the correct device via the zero3_init_context_manager() context manager. In particular, this is needed to avoid DeepSpeed hanging after a fixed number of training steps. Here is a snippet of what is involved from the sentiment_tuning example:

```python
ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
    with ds_plugin.zero3_init_context_manager(enable=False):
        sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
else:
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
```
