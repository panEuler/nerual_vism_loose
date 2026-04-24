from __future__ import annotations

import torch

from .area import _safe_query_grads, _stable_grad_norm


def eikonal_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    query_grads: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autograd-based eikonal penalty on masked batched query groups.
    
    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - mask: same leading shape as pred_sdf or None
    - query_points: [Q, 3] or [B, Q, 3]
    - query_grads: [Q, 3] or [B, Q, 3]
    
    
    """
    if mask is not None and not torch.any(mask):
        return pred_sdf.new_zeros(())

    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    penalty = (_stable_grad_norm(grads) - 1.0).abs()
    if mask is not None:
        penalty = penalty[mask]
    return penalty.mean()
