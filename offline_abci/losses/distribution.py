"""Distributional losses used by the augmentation network."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_distribution(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert arbitrary feature values to a valid batch-level distribution."""

    x = x.abs().mean(dim=0)
    return x / x.sum().clamp_min(eps)


def kl_divergence_loss(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute KL(p || q) for non-negative vectors."""

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    return (p * (p.log() - q.log())).sum()


def js_divergence_loss(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute Jensen-Shannon divergence."""

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence_loss(p, m, eps=eps) + 0.5 * kl_divergence_loss(q, m, eps=eps)


def hellinger_distance_loss(p: Tensor, q: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute the Hellinger distance between two distributions."""

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    return torch.linalg.vector_norm(torch.sqrt(p) - torch.sqrt(q)) / torch.sqrt(torch.tensor(2.0, device=p.device))


def correlation_distance_loss(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute one minus the Pearson correlation coefficient."""

    x = x.reshape(-1)
    y = y.reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    corr = (x * y).sum() / denom.clamp_min(eps)
    return 1.0 - corr


class DistributionLoss(nn.Module):
    """Combined distribution loss for generated feature batches."""

    def __init__(
        self,
        kl_weight: float = 100.0,
        js_weight: float = 100.0,
        hellinger_weight: float = 1.0,
        corr_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.js_weight = js_weight
        self.hellinger_weight = hellinger_weight
        self.corr_weight = corr_weight
        self.eps = eps

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """Compare generated and target batch-level feature distributions."""

        p = _as_distribution(generated, eps=self.eps)
        q = _as_distribution(target, eps=self.eps)
        return (
            self.kl_weight * kl_divergence_loss(p, q, eps=self.eps)
            + self.js_weight * js_divergence_loss(p, q, eps=self.eps)
            + self.hellinger_weight * hellinger_distance_loss(p, q, eps=self.eps)
            + self.corr_weight * correlation_distance_loss(generated.mean(dim=0), target.mean(dim=0), eps=self.eps)
        )
