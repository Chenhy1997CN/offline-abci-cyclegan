"""Trial-goal geometry utilities for feature-level augmentation."""

from __future__ import annotations

import torch
from torch import Tensor


def radial_scale_from_ratio(ratio: Tensor | float) -> Tensor:
    """Apply the radial scaling rule used by the trial-goal solver."""

    ratio_t = torch.as_tensor(ratio)
    return torch.where(ratio_t < 1.05, torch.tensor(0.9, device=ratio_t.device), 1.0 / ratio_t.clamp_min(1e-8))


def solve_trial_goal(
    source_trial_center: Tensor,
    target_center: Tensor,
    target_radius: Tensor | float,
    lr: float = 0.5,
    max_iter: int = 5000,
    tol: float = 1e-8,
) -> Tensor:
    """Solve a geometric trial-goal point by gradient-based optimization.

    The goal keeps the direction implied by ``source_trial_center`` while placing
    the point in a target-compatible radial region. This function only uses
    feature geometry and does not consume emotion labels.
    """

    dtype = source_trial_center.dtype
    device = source_trial_center.device
    source_trial_center = source_trial_center.detach().to(device=device, dtype=torch.float64)
    target_center = target_center.detach().to(device=device, dtype=torch.float64)
    target_radius_t = torch.as_tensor(target_radius, device=device, dtype=torch.float64).clamp_min(1e-8)

    source_radius = torch.linalg.vector_norm(source_trial_center - target_center).clamp_min(1e-8)
    ratio = source_radius / target_radius_t
    k = radial_scale_from_ratio(ratio).to(device=device, dtype=torch.float64)
    desired_radius = k * target_radius_t

    direction = (source_trial_center - target_center) / source_radius
    initial_goal = target_center + desired_radius * direction
    goal = initial_goal.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([goal], lr=lr)

    for _ in range(max_iter):
        optimizer.zero_grad()
        radial_residual = torch.linalg.vector_norm(goal - target_center) - desired_radius
        direction_residual = 1.0 - torch.nn.functional.cosine_similarity(
            (goal - target_center).unsqueeze(0),
            direction.unsqueeze(0),
        ).squeeze(0)
        loss = radial_residual.pow(2) + direction_residual.pow(2)
        if loss.item() < tol:
            break
        loss.backward()
        optimizer.step()
    return goal.detach().to(dtype=dtype)


def build_trial_goals(
    source_trial_centers: Tensor,
    target_center: Tensor,
    target_radius: Tensor | float,
    **solver_kwargs: object,
) -> Tensor:
    """Solve trial goals for a batch of source trial centers."""

    goals = [solve_trial_goal(center, target_center, target_radius, **solver_kwargs) for center in source_trial_centers]
    return torch.stack(goals, dim=0)
