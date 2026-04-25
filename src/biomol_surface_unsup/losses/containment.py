from __future__ import annotations

import torch


def containment_loss(
    pred_sdf: torch.Tensor,
    margin: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize points that are not sufficiently inside the predicted surface.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - mask: same leading shape as pred_sdf or None
    """
    penalty = torch.relu(pred_sdf + margin).pow(2)
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        penalty = penalty[mask]
    return penalty.mean()


def outside_loss(
    pred_sdf: torch.Tensor,
    margin: float = 0.5,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize points that are not sufficiently outside the predicted surface."""
    penalty = torch.relu(float(margin) - pred_sdf).pow(2)
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        penalty = penalty[mask]
    return penalty.mean()
