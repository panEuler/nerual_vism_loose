from __future__ import annotations

import torch


def smooth_heaviside(x: torch.Tensor, eps: float) -> torch.Tensor:
    eps = max(float(eps), 1e-6)
    return 0.5 * (1.0 + (2.0 / torch.pi) * torch.atan(x / eps))
