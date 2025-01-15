from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """
    model_name: Optional[str] = field(default="meta-llama/Llama-3.1-8B-Instruct", metadata={"help": "the model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./experiment",
        metadata={"help": "the path to save the model"},
    )


@dataclass
class GenerationKwargs:
    min_length: int
    #"top_k": 0.0,
    #"top_p": 1.0,
    num_beams: int
    temperature: float
    do_sample: bool
    pad_token_id: int
    max_new_tokens: int
