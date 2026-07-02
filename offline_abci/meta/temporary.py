"""Temporary meta-learning utilities for unknown-session adaptation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class TemporaryTask:
    """Pseudo-labeled temporary task for unknown-session adaptation."""

    features: Tensor
    pseudo_labels: Tensor
    trial_indices: Tensor
    reliability_scores: Tensor


def build_temporary_task(
    unknown_features: Tensor,
    trial_indices: Tensor,
    model_predictions: Tensor,
    reliability_scores: Tensor | None = None,
    top_k: int | None = None,
) -> TemporaryTask:
    """Build a temporary task from predictions without using ground-truth labels.

    Ground-truth labels of the unknown session must not be passed into this
    function. They should only be used after adaptation for final evaluation.
    """

    if unknown_features.ndim != 2:
        raise ValueError("unknown_features must have shape (num_samples, feature_dim).")
    trial_indices = trial_indices.to(device=unknown_features.device, dtype=torch.long).view(-1)
    if trial_indices.numel() != unknown_features.shape[0]:
        raise ValueError("trial_indices must have one entry per unknown sample.")

    probabilities = F.softmax(model_predictions.detach(), dim=1)
    confidence, pseudo_labels = probabilities.max(dim=1)
    pseudo_labels = pseudo_labels.to(device=unknown_features.device, dtype=torch.long)

    if reliability_scores is None:
        reliability_scores = torch.zeros_like(confidence)
        for trial_id in trial_indices.unique():
            mask = trial_indices == trial_id
            if not mask.any():
                continue
            labels_in_trial = pseudo_labels[mask]
            dominant_count = torch.bincount(labels_in_trial, minlength=model_predictions.shape[1]).max()
            stability = dominant_count.float() / labels_in_trial.numel()
            reliability_scores[mask] = confidence[mask] * stability
    else:
        reliability_scores = reliability_scores.to(device=unknown_features.device, dtype=unknown_features.dtype).view(-1)

    if top_k is not None:
        top_k = min(top_k, unknown_features.shape[0])
        selected = torch.topk(reliability_scores, k=top_k).indices
        unknown_features = unknown_features[selected]
        pseudo_labels = pseudo_labels[selected]
        trial_indices = trial_indices[selected]
        reliability_scores = reliability_scores[selected]

    return TemporaryTask(
        features=unknown_features,
        pseudo_labels=pseudo_labels,
        trial_indices=trial_indices,
        reliability_scores=reliability_scores,
    )


def temporary_meta_update(
    model: nn.Module,
    temporary_task: TemporaryTask,
    lr: float = 1e-6,
    epochs: int = 3,
    batch_size: int = 300,
    device: str | torch.device = "cuda",
) -> list[float]:
    """Adapt a model on a pseudo-labeled temporary task."""

    device = torch.device(device)
    model.to(device)
    model.train()
    features = temporary_task.features.to(device)
    labels = temporary_task.pseudo_labels.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.2, 0.999))
    losses: list[float] = []
    for _ in range(epochs):
        for start in range(0, features.shape[0], batch_size):
            end = min(start + batch_size, features.shape[0])
            optimizer.zero_grad()
            logits = model(features[start:end])
            loss = F.cross_entropy(logits, labels[start:end].long())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return losses
