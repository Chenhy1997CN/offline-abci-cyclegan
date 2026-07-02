"""CycleGAN components for EEG feature transfer and augmentation.

The models are intentionally lightweight fully connected networks because the
paper focuses on feature-level inter-subject data transfer rather than a new EEG
recognition backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


class Generator(nn.Module):
    """Fully connected generator used for DE feature transfer.

    Default architecture: 310 -> 256 -> 512 -> 1024 -> 512 -> 310.
    The input dimension 310 corresponds to 62 EEG channels times five
    frequency-band DE features.
    """

    def __init__(
        self,
        input_dim: int = 310,
        hidden_dims: tuple[int, int, int, int] = (256, 512, 1024, 512),
        negative_slope: float = 0.2,
        batch_norm_momentum: float = 0.8,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim, momentum=batch_norm_momentum))
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, input_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to all linear layers."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Map source-domain EEG features to a target-compatible space."""

        return self.net(x)


class Discriminator(nn.Module):
    """Fully connected discriminator for feature-domain discrimination."""

    def __init__(
        self,
        input_dim: int = 310,
        hidden_dims: tuple[int, int] = (512, 256),
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Apply Xavier initialization to all linear layers."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Return the probability that features come from the real domain."""

        return self.net(x)


@dataclass
class CycleGANOutputs:
    """Container for the main CycleGAN feature tensors."""

    ab: Tensor
    ba: Tensor
    aba: Tensor
    bab: Tensor


class CycleGANPair(nn.Module):
    """Bidirectional CycleGAN modules for domains A and B."""

    def __init__(self, feature_dim: int = 310) -> None:
        super().__init__()
        self.g_a2b = Generator(input_dim=feature_dim)
        self.g_b2a = Generator(input_dim=feature_dim)
        self.d_a = Discriminator(input_dim=feature_dim)
        self.d_b = Discriminator(input_dim=feature_dim)

    def transfer(self, a: Tensor, b: Tensor) -> CycleGANOutputs:
        """Compute transferred and cycle-reconstructed features."""

        ab = self.g_a2b(a)
        ba = self.g_b2a(b)
        aba = self.g_b2a(ab)
        bab = self.g_a2b(ba)
        return CycleGANOutputs(ab=ab, ba=ba, aba=aba, bab=bab)

    def forward(self, a: Tensor, b: Tensor) -> CycleGANOutputs:
        """Alias of :meth:`transfer` for standard PyTorch use."""

        return self.transfer(a, b)
