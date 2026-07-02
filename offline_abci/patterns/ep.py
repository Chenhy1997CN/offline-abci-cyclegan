"""Emotional Pattern (EP) matrix calculation."""

from __future__ import annotations

import torch
from torch import Tensor

from .centers import compute_class_centers, compute_class_dispersion


def _pairwise_center_distances(centers: Tensor) -> Tensor:
    """Return element-wise absolute distances for all class-center pairs."""

    pairs = []
    num_classes = centers.shape[0]
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            pairs.append((centers[i] - centers[j]).abs())
    if not pairs:
        return torch.empty(0, centers.shape[1], device=centers.device, dtype=centers.dtype)
    return torch.stack(pairs, dim=0)


def compute_ep_matrix(
    features: Tensor,
    labels: Tensor,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> Tensor:
    """Compute the EP matrix used by EP-constrained CycleGAN training.

    The matrix concatenates three feature-level descriptions: class centers,
    class-wise dispersion, and pairwise class-center distances. This function is
    differentiable with respect to ``features`` and can therefore be used in
    feature-generation losses.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim).")
    labels = labels.to(device=features.device, dtype=torch.long).view(-1)
    centers = compute_class_centers(features, labels, num_classes=num_classes)
    dispersion = compute_class_dispersion(features, labels, centers=centers, num_classes=num_classes, eps=eps)
    pairwise_distances = _pairwise_center_distances(centers)
    return torch.cat([centers, dispersion, pairwise_distances], dim=0)
