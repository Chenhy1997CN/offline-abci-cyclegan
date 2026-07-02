"""Simple trial grouping utilities for unknown-session features."""

from __future__ import annotations

import torch
from torch import Tensor


def group_trials_by_features(
    features: Tensor,
    num_trials: int | None = None,
    method: str = "sequential",
    random_state: int = 0,
) -> Tensor:
    """Estimate trial groups from feature samples.

    The default ``sequential`` method is deterministic and useful for synthetic
    tests or pre-segmented data. The optional ``kmeans`` method can be used when
    scikit-learn is available.
    """

    if features.ndim != 2:
        raise ValueError("features must have shape (num_samples, feature_dim).")
    num_samples = features.shape[0]
    if num_trials is None:
        num_trials = max(1, int(round(num_samples ** 0.5)))
    if method == "sequential":
        group_size = max(1, (num_samples + num_trials - 1) // num_trials)
        groups = torch.arange(num_samples, device=features.device) // group_size
        return groups.clamp_max(num_trials - 1).long()
    if method == "kmeans":
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError("scikit-learn is required for method='kmeans'.") from exc
        km = KMeans(n_clusters=num_trials, n_init=10, random_state=random_state)
        groups_np = km.fit_predict(features.detach().cpu().numpy())
        return torch.as_tensor(groups_np, device=features.device, dtype=torch.long)
    raise ValueError("method must be either 'sequential' or 'kmeans'.")
