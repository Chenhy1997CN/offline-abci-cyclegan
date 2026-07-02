# Offline-aBCI-CycleGAN

Core implementation for CycleGAN-based inter-subject EEG feature transfer/augmentation and WCMF-based meta-learning updates in offline affective brain-computer interfaces (aBCIs).

## Overview

This repository releases the main algorithmic components used in the associated study:

- CycleGAN generator, discriminator, and bidirectional transfer container.
- Emotional-pattern (EP) and correction T-test (CT) utilities.
- Reconstruction, augmentation, distribution, and Elastic-Net loss functions.
- Trial-goal geometry solver and trial grouping utilities.
- WCMF channel-wise classifier.
- Function-level recursive and temporary meta-learning updates.
- A synthetic test script for checking whether all public modules run correctly.

The code is provided as a lightweight library-style implementation. It does not include a complete experiment-running pipeline.

## Data

The experiments in the paper use SEED-series public EEG datasets. The raw datasets are not included in this repository. Please refer to the official SEED dataset website and apply for access according to the dataset policy:

https://bcmi.sjtu.edu.cn/home/seed/index.html#

## Project structure

```text
offline-abci-cyclegan/
  offline_abci/
    models/          # Generator, discriminator, WCMF
    patterns/        # EP, CT, center/radius utilities
    losses/          # Reconstruction, augmentation, distribution losses
    trial/           # Trial-goal solver and grouping utilities
    meta/            # Recursive and temporary meta-learning functions
    utils/           # Device and seed helpers
  tests/
    test_modules.py  # Synthetic functionality test
```

## Installation

The manuscript experiments were implemented with PyTorch 2.10.0.

```bash
pip install -r requirements.txt
```

Optional editable installation:

```bash
pip install -e .
```

## Module test

Run the synthetic test without any real EEG dataset:

```bash
python tests/test_modules.py
```

Expected output:

```text
All modules passed the synthetic functionality test.
```

## Minimal usage

```python
import torch

from offline_abci.models.gan import Generator, Discriminator
from offline_abci.models.wcmf import WCMF
from offline_abci.patterns.ep import compute_ep_matrix
from offline_abci.patterns.ct import compute_ct_weights

x = torch.randn(75, 310)
y = torch.arange(75) % 3

G = Generator(input_dim=310)
D = Discriminator(input_dim=310)
wcmf = WCMF(num_channels=62, num_bands=5, num_classes=3)

x_transfer = G(x)
d_score = D(x_transfer)
logits = wcmf(x)

ep = compute_ep_matrix(x, y, num_classes=3)
weight, corr_weight, emotion_type = compute_ct_weights(x, y, num_classes=3)
```

## Citation

If you use this code, please cite the associated manuscript:

```text
Huayu Chen et al., "An Offline Affective Brain-Computer Interface Paradigm with CycleGAN Data Augmentation and Meta-Learning Update Mode."
```

The citation metadata can be updated after the paper is formally published.

## License

Please confirm the final license choice with all authors and the affiliated institution before public release.
