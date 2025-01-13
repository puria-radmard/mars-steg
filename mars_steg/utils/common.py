import torch as t
import numpy as np
import random

def get_device() -> str:
    if t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    #if t.cuda.is_available() and t.cuda.device_count() > 1:
        #raise ValueError("Please specify which GPU to use with CUDA_VISIBLE_DEVICES")

    return device

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False