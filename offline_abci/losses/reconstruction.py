"""Losses for EP/CT-constrained information reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from offline_abci.patterns.centers import compute_class_medians
from offline_abci.patterns.ct import CTWeights, compute_ct_weights
from offline_abci.patterns.ep import compute_ep_matrix


@dataclass
class ReconstructionLossConfig:
    """Weights used by the information reconstruction objective."""

    ep_weight: float = 1.0
    ct_weight: float = 1.0
    centroid_weight: float = 10.0
    cycle_weight: float = 1.0


class EPLoss(nn.Module):
    """CT-weighted element-wise L1 loss between EP matrices."""

    def forward(self, reference_ep: Tensor, generated_ep: Tensor, feature_weight: Tensor | None = None) -> Tensor:
        loss = F.l1_loss(generated_ep, reference_ep, reduction="none")
        if feature_weight is not None:
            while feature_weight.ndim < loss.ndim:
                feature_weight = feature_weight.unsqueeze(0)
            loss = loss * feature_weight.to(device=loss.device, dtype=loss.dtype)
        return loss.mean()


class CTConsistencyLoss(nn.Module):
    """Consistency loss between CT weights and CT-derived emotion-type matrices."""

    def forward(
        self,
        reference: CTWeights,
        generated: CTWeights,
        feature_weight: Tensor | None = None,
    ) -> Tensor:
        mse = F.mse_loss(generated.corrected_weight, reference.corrected_weight)
        type_loss = F.l1_loss(generated.emotion_type, reference.emotion_type, reduction="none")
        if feature_weight is not None:
            while feature_weight.ndim < type_loss.ndim:
                feature_weight = feature_weight.unsqueeze(0)
            type_loss = type_loss * feature_weight.to(device=type_loss.device, dtype=type_loss.dtype)
        return mse + type_loss.mean()


class CentroidReconstructionLoss(nn.Module):
    """Pull transferred/reconstructed samples to class-wise median regions."""

    def forward(self, reference_features: Tensor, reference_labels: Tensor, generated: Tensor, generated_labels: Tensor) -> Tensor:
        medians = compute_class_medians(reference_features, reference_labels)
        generated_labels = generated_labels.to(device=generated.device, dtype=torch.long).view(-1)
        target_medians = medians[generated_labels]
        return F.l1_loss(generated, target_medians)


class CycleConsistencyLoss(nn.Module):
    """Standard feature-level cycle-consistency loss."""

    def forward(self, original: Tensor, reconstructed: Tensor) -> Tensor:
        return F.l1_loss(reconstructed, original)


def reconstruction_step_losses(
    a: Tensor,
    b: Tensor,
    labels_a: Tensor,
    labels_b: Tensor,
    ab: Tensor,
    ba: Tensor,
    aba: Tensor,
    bab: Tensor,
    config: ReconstructionLossConfig | None = None,
) -> dict[str, Tensor]:
    """Compute the main losses for one information-reconstruction step.

    The function exposes the core objective without assuming a particular
    training script. It can be used inside custom training loops.
    """

    if config is None:
        config = ReconstructionLossConfig()

    ep_loss_fn = EPLoss()
    ct_loss_fn = CTConsistencyLoss()
    centroid_loss_fn = CentroidReconstructionLoss()
    cycle_loss_fn = CycleConsistencyLoss()

    ct_a = compute_ct_weights(a, labels_a)
    ct_b = compute_ct_weights(b, labels_b)
    ct_ab = compute_ct_weights(ab, labels_a)
    ct_ba = compute_ct_weights(ba, labels_b)
    ct_aba = compute_ct_weights(aba, labels_a)
    ct_bab = compute_ct_weights(bab, labels_b)

    ep_a = compute_ep_matrix(a, labels_a)
    ep_b = compute_ep_matrix(b, labels_b)
    loss_ep = (
        ep_loss_fn(ep_a, compute_ep_matrix(aba, labels_a), ct_a.weight)
        + ep_loss_fn(ep_a, compute_ep_matrix(ba, labels_b), ct_a.weight)
        + ep_loss_fn(ep_b, compute_ep_matrix(bab, labels_b), ct_b.weight)
        + ep_loss_fn(ep_b, compute_ep_matrix(ab, labels_a), ct_b.weight)
    )

    loss_ct = (
        ct_loss_fn(ct_a, ct_aba, ct_a.weight)
        + ct_loss_fn(ct_a, ct_ba, ct_a.weight)
        + ct_loss_fn(ct_b, ct_bab, ct_b.weight)
        + ct_loss_fn(ct_b, ct_ab, ct_b.weight)
    )

    loss_centroid = (
        centroid_loss_fn(a, labels_a, aba, labels_a)
        + centroid_loss_fn(a, labels_a, ba, labels_b)
        + centroid_loss_fn(b, labels_b, bab, labels_b)
        + centroid_loss_fn(b, labels_b, ab, labels_a)
    )
    loss_cycle = cycle_loss_fn(a, aba) + cycle_loss_fn(b, bab)

    total = (
        config.ep_weight * loss_ep
        + config.ct_weight * loss_ct
        + config.centroid_weight * loss_centroid
        + config.cycle_weight * loss_cycle
    )
    return {
        "ep": loss_ep,
        "ct": loss_ct,
        "centroid": loss_centroid,
        "cycle": loss_cycle,
        "total": total,
    }
