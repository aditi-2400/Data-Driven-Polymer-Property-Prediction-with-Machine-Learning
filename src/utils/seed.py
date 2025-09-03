import os, random, numpy as np

def set_global_seed(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)

    except Exception:
        pass
    np.random.seed(seed)
