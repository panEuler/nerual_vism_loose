from __future__ import annotations

import torch

from biomol_surface_unsup.geometry.sdf_ops import box_sdf
from biomol_surface_unsup.utils.pairwise import chunked_smooth_atomic_union_field


def _batched_atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    return chunked_smooth_atomic_union_field(coords, radii, query_points)


def weak_prior_loss(
    coords: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    atom_mask: torch.Tensor | None = None,
    target_type: str = "atomic_union",
    bbox_lower: torch.Tensor | None = None,
    bbox_upper: torch.Tensor | None = None,
) -> torch.Tensor:
    """Toy weak prior against the atomic-union proxy.
    
    This acts as a geometric bootstrap. Without it, unsupervised physics losses might 
    initially trap the network into trivial minimums (e.g., shrinking to an empty vacuum).
    This loss pulls the initial network shape toward the basic Van der Waals surface,
    until other fine-grained physics losses dominate and take over.

    Batched shapes:
    - coords: [B, N, 3]
    - radii: [B, N]
    - query_points: [B, Q, 3]
    - pred_sdf: [B, Q]
    - mask: [B, Q] or None
    - atom_mask: [B, N] or None
    """
    # Handle single sample tensors by temporarily unsqueezing a pseudo-batch dimension
    squeeze_batch = coords.ndim == 2
    if squeeze_batch:
        coords = coords.unsqueeze(0)
        radii = radii.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        pred_sdf = pred_sdf.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)
        atom_mask = None if atom_mask is None else atom_mask.unsqueeze(0)

    target_name = str(target_type).lower()
    if target_name in {"none", "off", "disabled"}:
        return pred_sdf.new_zeros(())
    if target_name not in {"atomic_union", "box"}:
        raise ValueError("weak_prior target_type must be one of: atomic_union, box, none")

    if target_name == "atomic_union":
        # Ensure atom masks are present to avoid considering tensor padding zeros as real atomic structural data.
        if atom_mask is None:
            atom_mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)

        # Mask out padded invalid atoms by zeroing out their radius and coordinates.
        safe_radii = radii.masked_fill(~atom_mask, 0.0)
        safe_coords = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)

        # Generate the geometric proxy baseline for supervision.
        # `.detach()` is crucial here. It stops the gradient graph so this proxy acts strictly as fixed target labels.
        with torch.no_grad():
            target = _batched_atomic_union_field(safe_coords, safe_radii, query_points)
    else:
        if bbox_lower is None or bbox_upper is None:
            raise ValueError("bbox_lower and bbox_upper are required for box weak_prior target")
        with torch.no_grad():
            target = box_sdf(query_points, bbox_lower, bbox_upper)

    # Calculate L1 Loss forcing the network to match the rough VdW target shape.
    if mask is not None:
        if not torch.any(mask):
            return pred_sdf.new_zeros(())
        # Only compute the mean absolute error on valid, unpadded spatial query points.
        return (pred_sdf[mask] - target[mask]).abs().mean()
        
    return (pred_sdf - target).abs().mean()
