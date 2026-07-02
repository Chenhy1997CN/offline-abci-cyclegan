"""Function-level recursive meta-learning utilities.

The functions in this module operate on prepared tensor tasks. They deliberately
avoid dataset paths and full experiment orchestration so they can be embedded in
custom training protocols.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TensorTask = tuple[Tensor, Tensor, Tensor, Tensor]


@dataclass
class MetaTrainHistory:
    """Summary of recursive meta-learning updates."""

    query_losses: list[float]
    query_accuracies: list[float]


def _iterate_tensor_batches(x: Tensor, y: Tensor, batch_size: int) -> Iterable[tuple[Tensor, Tensor]]:
    """Yield mini-batches from tensor data without assuming a DataLoader."""

    num_samples = x.shape[0]
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        yield x[start:end], y[start:end]


def _train_inner_model(
    model: nn.Module,
    support_x: Tensor,
    support_y: Tensor,
    inner_lr: float,
    inner_epochs: int,
    batch_size: int,
) -> nn.Module:
    """Adapt a copy of the base model on support data."""

    adapted = copy.deepcopy(model)
    adapted.train()
    optimizer = torch.optim.Adam(adapted.parameters(), lr=inner_lr, betas=(0.2, 0.999))
    for _ in range(inner_epochs):
        for batch_x, batch_y in _iterate_tensor_batches(support_x, support_y, batch_size):
            optimizer.zero_grad()
            logits = adapted(batch_x)
            loss = F.cross_entropy(logits, batch_y.long())
            loss.backward()
            optimizer.step()
    return adapted


def _evaluate_query(model: nn.Module, query_x: Tensor, query_y: Tensor) -> tuple[Tensor, float]:
    """Evaluate the adapted model on query data."""

    logits = model(query_x)
    loss = F.cross_entropy(logits, query_y.long())
    acc = (logits.argmax(dim=1) == query_y.long()).float().mean().item()
    return loss, acc


def recursive_meta_train(
    model: nn.Module,
    tasks: list[TensorTask],
    inner_lr: float = 1e-6,
    outer_lr: float = 1e-7,
    inner_epochs: int = 3,
    outer_epochs: int = 1,
    batch_size: int = 300,
    device: str | torch.device = "cuda",
) -> MetaTrainHistory:
    """Perform a lightweight first-order recursive meta-learning update.

    For each task, a copied model is adapted on support data. The base model is
    then moved slightly toward the adapted parameters. This implementation keeps
    the update transparent and stable for module-level release while preserving
    the inner/outer update structure.
    """

    device = torch.device(device)
    model.to(device)
    history = MetaTrainHistory(query_losses=[], query_accuracies=[])

    for _ in range(outer_epochs):
        for support_x, support_y, query_x, query_y in tasks:
            support_x = support_x.to(device)
            support_y = support_y.to(device)
            query_x = query_x.to(device)
            query_y = query_y.to(device)

            adapted = _train_inner_model(
                model,
                support_x,
                support_y,
                inner_lr=inner_lr,
                inner_epochs=inner_epochs,
                batch_size=batch_size,
            )
            with torch.no_grad():
                query_loss, query_acc = _evaluate_query(adapted, query_x, query_y)
                history.query_losses.append(float(query_loss.item()))
                history.query_accuracies.append(float(query_acc))
                # First-order outer update: move base parameters toward adapted parameters.
                for base_param, adapted_param in zip(model.parameters(), adapted.parameters()):
                    base_param.add_(outer_lr * (adapted_param - base_param))
    return history
