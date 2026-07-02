"""Device helpers."""

from __future__ import annotations

import torch


def get_default_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when available, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
