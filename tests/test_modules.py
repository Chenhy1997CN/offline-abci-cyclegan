"""Synthetic functionality test for all public modules.

This script does not require SEED, SEED-FRA, or SEED-GER data. It only verifies
that model forward passes, pattern calculations, losses, trial utilities, and
meta-learning updates run correctly on synthetic EEG-like DE features.
"""

from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

# Keep synthetic tests fast and deterministic on shared CPU environments.
torch.set_num_threads(1)

# Allow running this file directly without installing the package first.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from offline_abci.losses import AugmentationLoss, elastic_net_regularization, reconstruction_step_losses
from offline_abci.losses.distribution import DistributionLoss
from offline_abci.meta import build_temporary_task, recursive_meta_train, temporary_meta_update
from offline_abci.models import CycleGANPair, Discriminator, Generator, WCMF
from offline_abci.patterns import compute_ct_weights, compute_domain_center, compute_domain_radius, compute_ep_matrix
from offline_abci.trial import group_trials_by_features, solve_trial_goal
from offline_abci.utils import get_default_device, set_seed


def make_synthetic_batch(
    batch_size: int = 24,
    feature_dim: int = 310,
    num_classes: int = 3,
    domain_shift: float = 0.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a balanced synthetic EEG-like feature batch."""

    x = torch.randn(batch_size, feature_dim, device=device) + domain_shift
    y = torch.arange(batch_size, device=device) % num_classes
    # Add mild class structure to avoid degenerate class statistics.
    for class_idx in range(num_classes):
        x[y == class_idx, class_idx::num_classes] += 0.2 * class_idx
    return x, y.long()


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise a clear error if a tensor contains NaN or inf values."""

    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def test_gan_modules(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Test generator, discriminator, and CycleGANPair outputs."""

    a, labels_a = make_synthetic_batch(device=device)
    b, labels_b = make_synthetic_batch(domain_shift=0.5, device=device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    gan = CycleGANPair().to(device)

    generated = generator(a)
    score = discriminator(a)
    outputs = gan(a, b)

    assert generated.shape == a.shape
    assert score.shape == (a.shape[0], 1)
    assert outputs.ab.shape == a.shape
    assert outputs.ba.shape == b.shape
    assert outputs.aba.shape == a.shape
    assert outputs.bab.shape == b.shape
    return a, b, labels_a, labels_b, outputs.ab, outputs.ba


def test_patterns_and_reconstruction_loss(device: torch.device) -> None:
    """Test EP/CT calculations and reconstruction losses."""

    a, labels_a = make_synthetic_batch(device=device)
    b, labels_b = make_synthetic_batch(domain_shift=0.5, device=device)
    gan = CycleGANPair().to(device)
    outputs = gan(a, b)

    ep = compute_ep_matrix(a, labels_a)
    ct = compute_ct_weights(a, labels_a)
    assert ep.shape[1] == a.shape[1]
    assert ct.weight.shape == (a.shape[1],)
    assert ct.corrected_weight.ndim == 2

    loss_dict = reconstruction_step_losses(
        a,
        b,
        labels_a,
        labels_b,
        outputs.ab,
        outputs.ba,
        outputs.aba,
        outputs.bab,
    )
    assert set(loss_dict) == {"ep", "ct", "centroid", "cycle", "total"}
    assert_finite("reconstruction total loss", loss_dict["total"])
    loss_dict["total"].backward()


def test_augmentation_and_distribution_loss(device: torch.device) -> None:
    """Test augmentation losses and distribution loss."""

    generated, labels_g = make_synthetic_batch(device=device)
    target, labels_t = make_synthetic_batch(domain_shift=0.25, device=device)
    trial_goals = generated.detach() * 0.95 + target.mean(dim=0, keepdim=True) * 0.05

    augmentation_loss = AugmentationLoss().to(device)
    loss_dict = augmentation_loss(generated, target, labels_g, labels_t, trial_goals=trial_goals)
    assert_finite("augmentation total loss", loss_dict["total"])

    distribution_loss = DistributionLoss().to(device)
    dist_loss = distribution_loss(generated, target)
    assert_finite("distribution loss", dist_loss)


def test_trial_utilities(device: torch.device) -> None:
    """Test trial grouping and trial-goal solver."""

    features, _ = make_synthetic_batch(batch_size=18, device=device)
    groups = group_trials_by_features(features, num_trials=6, method="sequential")
    assert groups.shape == (features.shape[0],)
    assert groups.max().item() <= 5

    center = compute_domain_center(features)
    radius = compute_domain_radius(features, center)
    goal = solve_trial_goal(features[0], center, radius, max_iter=5)
    assert goal.shape == center.shape
    assert_finite("trial goal", goal)


def test_wcmf_and_meta_learning(device: torch.device) -> None:
    """Test WCMF forward pass, recursive update, and temporary update."""

    x1, y1 = make_synthetic_batch(batch_size=18, feature_dim=30, device=device)
    x2, y2 = make_synthetic_batch(batch_size=18, feature_dim=30, domain_shift=0.2, device=device)
    x3, y3 = make_synthetic_batch(batch_size=18, feature_dim=30, domain_shift=-0.2, device=device)
    x4, y4 = make_synthetic_batch(batch_size=18, feature_dim=30, domain_shift=0.4, device=device)

    model = WCMF(num_channels=6, num_bands=5).to(device)
    logits = model(x1)
    assert logits.shape == (x1.shape[0], 3)
    loss = F.cross_entropy(logits, y1)
    loss.backward()

    # Keep test updates short; the function defaults correspond to the paper settings.
    tasks = [(x1, y1, x2, y2), (x3, y3, x4, y4)]
    history = recursive_meta_train(
        model,
        tasks,
        inner_lr=1e-4,
        outer_lr=1e-2,
        inner_epochs=1,
        outer_epochs=1,
        batch_size=9,
        device=device,
    )
    assert len(history.query_losses) == 2

    with torch.no_grad():
        unknown_logits = model(x4)
    trial_indices = torch.arange(x4.shape[0], device=device) // 5
    temporary_task = build_temporary_task(x4, trial_indices, unknown_logits, top_k=20)
    update_losses = temporary_meta_update(
        model,
        temporary_task,
        lr=1e-4,
        epochs=1,
        batch_size=9,
        device=device,
    )
    assert update_losses

    reg = elastic_net_regularization(model)
    assert_finite("elastic net regularization", reg)


def main() -> None:
    """Run all synthetic module tests."""

    set_seed(42)
    device = get_default_device()
    print(f"Running synthetic module tests on {device}.")

    test_gan_modules(device)
    test_patterns_and_reconstruction_loss(device)
    test_augmentation_and_distribution_loss(device)
    test_trial_utilities(device)
    test_wcmf_and_meta_learning(device)

    print("All modules passed the synthetic functionality test.")


if __name__ == "__main__":
    main()
