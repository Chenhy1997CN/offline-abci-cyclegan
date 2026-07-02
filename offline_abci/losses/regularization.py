"""Regularization utilities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def elastic_net_regularization(
    model: nn.Module,
    lambda_l1: float = 0.01,
    lambda_l2: float = 0.99,
) -> Tensor:
    """Compute Elastic-Net regularization for all trainable parameters."""

    device = next(model.parameters()).device
    reg = torch.zeros((), device=device)
    for param in model.parameters():
        if param.requires_grad:
            reg = reg + lambda_l1 * param.abs().sum() + lambda_l2 * param.pow(2).sum()
    return reg
