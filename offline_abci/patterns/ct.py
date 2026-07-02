"""Correction T-test (CT) style feature-weight calculation.

The original project used correction T-test weights to highlight emotion-related
feature dimensions and their directions. This module provides a robust PyTorch
implementation based on Welch-style pairwise class statistics. It avoids hard
SciPy dependencies inside the training graph while preserving the role of CT
weights as feature-level compatibility constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class CTWeights:
    """Container for CT-related statistics."""

    weight: Tensor
    corrected_weight: Tensor
    emotion_type: Tensor


def _class_stats(features: Tensor, labels: Tensor, num_classes: int, eps: float) -> tuple[Tensor, Tensor, Tensor]:
    """Return class means, variances, and counts."""

    means = []
    variances = []
    counts = []
    for class_idx in range(num_classes):
        class_features = features[labels == class_idx]
        if class_features.numel() == 0:
            means.append(torch.zeros(features.shape[1], device=features.device, dtype=features.dtype))
            variances.append(torch.ones(features.shape[1], device=features.device, dtype=features.dtype))
            counts.append(torch.tensor(1.0, device=features.device, dtype=features.dtype))
        else:
            means.append(class_features.mean(dim=0))
            variances.append(class_features.var(dim=0, unbiased=False).clamp_min(eps))
            counts.append(torch.tensor(float(class_features.shape[0]), device=features.device, dtype=features.dtype))
    return torch.stack(means), torch.stack(variances), torch.stack(counts)


def compute_ct_weights(
    features: Tensor,
    labels: Tensor,
    num_classes: int = 3,
    eps: float = 1e-8,
) -> CTWeights:
    """Compute CT-style feature weights and direction patterns.

    Returns
    -------
    CTWeights
        ``weight`` has shape ``(feature_dim,)`` and indicates the global
        importance of each feature dimension. ``corrected_weight`` and
        ``emotion_type`` have shape ``(num_pairs, feature_dim)`` and retain
        pairwise class-direction information.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim).")
    labels = labels.to(device=features.device, dtype=torch.long).view(-1)
    means, variances, counts = _class_stats(features, labels, num_classes, eps)

    corrected = []
    emotion_types = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mean_diff = means[i] - means[j]
            pooled_se = torch.sqrt(variances[i] / counts[i] + variances[j] / counts[j] + eps)
            t_score = mean_diff / pooled_se
            # Smoothly compress t-scores to a stable range while preserving direction.
            corrected.append(torch.tanh(t_score))
            emotion_types.append(torch.sign(mean_diff))

    corrected_weight = torch.stack(corrected, dim=0)
    emotion_type = torch.stack(emotion_types, dim=0)
    weight = corrected_weight.abs().mean(dim=0).clamp_min(eps)
    weight = weight / weight.max().clamp_min(eps)
    return CTWeights(weight=weight, corrected_weight=corrected_weight, emotion_type=emotion_type)
