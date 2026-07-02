"""Losses for the data augmentation network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from offline_abci.losses.distribution import DistributionLoss
from offline_abci.patterns.centers import (
    compute_class_centers,
    compute_domain_center,
    compute_domain_radius,
)


@dataclass
class AugmentationLossConfig:
    """Weights used by the augmentation objective."""

    trial_goal_weight: float = 10.0
    center_weight: float = 100.0
    radius_weight: float = 100.0
    class_center_weight: float = 100.0
    distribution_weight: float = 1.0


class TrialGoalLoss(nn.Module):
    """MSE loss between augmented samples and trial-goal references."""

    def forward(self, generated: Tensor, trial_goals: Tensor) -> Tensor:
        if generated.shape != trial_goals.shape:
            raise ValueError("generated and trial_goals must have identical shapes.")
        return F.mse_loss(generated, trial_goals)


class TargetCenterLoss(nn.Module):
    """MSE loss between generated and target domain centers."""

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(compute_domain_center(generated), compute_domain_center(target))


class RadiusLoss(nn.Module):
    """Match the global radius of generated and target domains."""

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        gen_radius = compute_domain_radius(generated)
        target_radius = compute_domain_radius(target)
        return F.mse_loss(gen_radius, target_radius)


class ClassCenterLoss(nn.Module):
    """MSE loss between generated and target class centers."""

    def forward(
        self,
        generated: Tensor,
        generated_labels: Tensor,
        target: Tensor,
        target_labels: Tensor,
        num_classes: int = 3,
    ) -> Tensor:
        gen_centers = compute_class_centers(generated, generated_labels, num_classes=num_classes)
        target_centers = compute_class_centers(target, target_labels, num_classes=num_classes)
        return F.mse_loss(gen_centers, target_centers)


class AugmentationLoss(nn.Module):
    """Combined augmentation loss.

    This module only receives tensors and can be used inside any user-defined
    training loop. It does not assume a specific dataset layout.
    """

    def __init__(self, config: AugmentationLossConfig | None = None, num_classes: int = 3) -> None:
        super().__init__()
        self.config = config or AugmentationLossConfig()
        self.num_classes = num_classes
        self.trial_goal_loss = TrialGoalLoss()
        self.center_loss = TargetCenterLoss()
        self.radius_loss = RadiusLoss()
        self.class_center_loss = ClassCenterLoss()
        self.distribution_loss = DistributionLoss()

    def forward(
        self,
        generated: Tensor,
        target: Tensor,
        generated_labels: Tensor,
        target_labels: Tensor,
        trial_goals: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute all augmentation losses and return a named dictionary."""

        zero = torch.zeros((), device=generated.device, dtype=generated.dtype)
        loss_trial = self.trial_goal_loss(generated, trial_goals) if trial_goals is not None else zero
        loss_center = self.center_loss(generated, target)
        loss_radius = self.radius_loss(generated, target)
        loss_class = self.class_center_loss(
            generated,
            generated_labels,
            target,
            target_labels,
            num_classes=self.num_classes,
        )
        loss_distribution = self.distribution_loss(generated, target)
        total = (
            self.config.trial_goal_weight * loss_trial
            + self.config.center_weight * loss_center
            + self.config.radius_weight * loss_radius
            + self.config.class_center_weight * loss_class
            + self.config.distribution_weight * loss_distribution
        )
        return {
            "trial_goal": loss_trial,
            "center": loss_center,
            "radius": loss_radius,
            "class_center": loss_class,
            "distribution": loss_distribution,
            "total": total,
        }
