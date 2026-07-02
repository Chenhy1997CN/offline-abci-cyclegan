"""Weight-based Channel-model Matrix Framework (WCMF).

WCMF keeps one lightweight classifier for each EEG channel. Each channel model
receives five frequency-band features and produces class logits. Channel-wise
outputs are then weighted and summed for the final prediction.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelModel(nn.Module):
    """Three-layer channel classifier: 5 -> 15 -> 10 -> num_classes."""

    def __init__(
        self,
        num_bands: int = 5,
        num_classes: int = 3,
        hidden_dims: tuple[int, int] = (15, 10),
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bands, hidden_dims[0]),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Linear(hidden_dims[1], num_classes),
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
        """Return channel-level class logits."""

        return self.net(x)


class WCMF(nn.Module):
    """A matrix of channel models with channel-emotion weights.

    Parameters
    ----------
    num_channels:
        Number of EEG channels. The paper uses 62 channels.
    num_bands:
        Number of DE frequency bands. The paper uses five bands.
    num_classes:
        Number of emotion classes.
    normalize_weights:
        If true, channel weights are normalized across channels before summing.
    """

    def __init__(
        self,
        num_channels: int = 62,
        num_bands: int = 5,
        num_classes: int = 3,
        normalize_weights: bool = True,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.num_classes = num_classes
        self.normalize_weights = normalize_weights
        self.channel_models = nn.ModuleList(
            [ChannelModel(num_bands=num_bands, num_classes=num_classes) for _ in range(num_channels)]
        )
        # Shape: (channels, classes). The default gives all channels equal weight.
        self.channel_weights = nn.Parameter(torch.ones(num_channels, num_classes))

    def _reshape_input(self, x: Tensor) -> Tensor:
        """Accept either flattened (B, 310) or structured (B, 62, 5) features."""

        if x.ndim == 2:
            expected_dim = self.num_channels * self.num_bands
            if x.shape[1] != expected_dim:
                raise ValueError(f"Expected flattened feature_dim={expected_dim}, got {x.shape[1]}.")
            return x.view(x.shape[0], self.num_channels, self.num_bands)
        if x.ndim == 3:
            if x.shape[1:] != (self.num_channels, self.num_bands):
                raise ValueError(
                    f"Expected shape (batch, {self.num_channels}, {self.num_bands}), got {tuple(x.shape)}."
                )
            return x
        raise ValueError("WCMF input must have shape (batch, 310) or (batch, 62, 5).")

    def forward_channels(self, x: Tensor) -> Tensor:
        """Return logits for all channel models with shape (B, channels, classes)."""

        x = self._reshape_input(x)
        logits = []
        for channel_idx, channel_model in enumerate(self.channel_models):
            logits.append(channel_model(x[:, channel_idx, :]))
        return torch.stack(logits, dim=1)

    def normalized_channel_weights(self) -> Tensor:
        """Return non-negative channel weights used to aggregate channel logits."""

        weights = torch.relu(self.channel_weights)
        if self.normalize_weights:
            weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-8)
        return weights

    def set_channel_weights(self, weights: Tensor, trainable: bool = True) -> None:
        """Set channel-class weights from prior EP/CT information.

        ``weights`` can have shape (channels,), (channels, classes), or
        flattened shape (channels * classes,). Values are copied into the model.
        """

        with torch.no_grad():
            weights = weights.to(self.channel_weights.device, dtype=self.channel_weights.dtype)
            if weights.ndim == 1 and weights.numel() == self.num_channels:
                weights = weights[:, None].repeat(1, self.num_classes)
            elif weights.ndim == 1 and weights.numel() == self.num_channels * self.num_classes:
                weights = weights.view(self.num_channels, self.num_classes)
            elif weights.shape != self.channel_weights.shape:
                raise ValueError(f"Invalid weight shape {tuple(weights.shape)}.")
            self.channel_weights.copy_(weights)
        self.channel_weights.requires_grad_(trainable)

    def forward(self, x: Tensor) -> Tensor:
        """Aggregate channel-wise logits into final class logits."""

        channel_logits = self.forward_channels(x)
        weights = self.normalized_channel_weights().unsqueeze(0)
        return (channel_logits * weights).sum(dim=1)
