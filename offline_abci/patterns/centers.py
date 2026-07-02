"""Feature center, median, and radius utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_features_and_labels(features: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Validate and standardize feature and label tensors."""

    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim).")
    labels = labels.to(device=features.device, dtype=torch.long).view(-1)
    if labels.numel() != features.shape[0]:
        raise ValueError("labels must have one entry per sample.")
    return features, labels


def compute_class_centers(features: Tensor, labels: Tensor, num_classes: int = 3) -> Tensor:
    """Compute class-wise feature centers.

    Empty classes are assigned a zero vector. In balanced EEG mini-batches this
    should not occur, but the fallback keeps synthetic tests robust.
    """

    features, labels = _validate_features_and_labels(features, labels)
    centers = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.any():
            centers.append(features[mask].mean(dim=0))
        else:
            centers.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
    return torch.stack(centers, dim=0)


def compute_class_medians(features: Tensor, labels: Tensor, num_classes: int = 3) -> Tensor:
    """Compute class-wise feature medians."""

    features, labels = _validate_features_and_labels(features, labels)
    medians = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.any():
            medians.append(features[mask].median(dim=0).values)
        else:
            medians.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
    return torch.stack(medians, dim=0)


def compute_class_dispersion(
    features: Tensor,
    labels: Tensor,
    centers: Tensor | None = None,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> Tensor:
    """Compute mean absolute dispersion around each class center."""

    features, labels = _validate_features_and_labels(features, labels)
    if centers is None:
        centers = compute_class_centers(features, labels, num_classes=num_classes)
    dispersions = []
    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.any():
            dispersions.append((features[mask] - centers[class_idx]).abs().mean(dim=0))
        else:
            dispersions.append(torch.full((features.shape[1],), eps, device=features.device, dtype=features.dtype))
    return torch.stack(dispersions, dim=0)


def compute_domain_center(features: Tensor) -> Tensor:
    """Compute the global center of a feature domain."""

    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim).")
    return features.mean(dim=0)


def compute_domain_radius(features: Tensor, center: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    """Compute the average Euclidean radius around the domain center."""

    if center is None:
        center = compute_domain_center(features)
    return torch.linalg.vector_norm(features - center, dim=1).mean().clamp_min(eps)
