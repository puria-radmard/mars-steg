# Running Sweeps

Typically you want to run this on a multi-GPU setup. When you get onto your GPU first log in to wandb with:

```bash
wandb login
```

This will prompt you to add your credentials so you can track the run.

Then initialise the project repo so it appears in the correct place with:

```bash
wandb init
```

Now retrieve the sweep id and start the sweep by running:

```bash
wandb sweep sweeps/[SWEEP_NAME].yaml
```

This will return a `SWEEP_ID` you can use for sweeps. Note that although your sweep is initialised, nothing is running yet.

First run `nvidia-smi` to check which GPUs are available.

To actually run the sweep write:

```bash
CUDA_VISIBLE_DEVICES=0 wandb agent [SWEEP_ID]
...
CUDA_VISIBLE_DEVICES=n wandb agent [SWEEP_ID]
```

or (need to test)

```bash
CUDA_VISIBLE_DEVICES=0,1 wandb agent [SWEEP_ID]
...
CUDA_VISIBLE_DEVICES=n-1,n wandb agent [SWEEP_ID]
```

For each GPU you have available.

Or on a single GPU machine, simply run:

```bash
wandb agent [SWEEP_ID]
```

Now go and look in the wandb console to see how your sweep is going.
You can also pause and kill sweeps from the UI console.

To learn more see the docs [here](https://docs.wandb.ai/guides/sweeps)

To move a process into the background use:

1. Suspend the process: Press `Ctrl+Z` in the terminal where the process is running. This will suspend the process.
2. Move it to the background: Type `bg` and press Enter. This resumes the process in the background.
3. Disown the process: Type `disown` and press Enter. This detaches the process from the terminal.