from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_global_seed(seed: int, deterministic_torch: bool = True) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return seed
