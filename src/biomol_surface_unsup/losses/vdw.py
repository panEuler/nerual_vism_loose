from __future__ import annotations

import torch

from biomol_surface_unsup.utils.pairwise import chunked_lj_potential_sum

from .area import _masked_monte_carlo_integral
from .heaviside import smooth_heaviside


def lj_body_integral(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    coords: torch.Tensor,
    epsilon_lj: torch.Tensor,
    sigma_lj: torch.Tensor,
    atom_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    rho_0: float = 0.0334,
    eps_h: float = 0.1,
    dist_eps: float = 1.5,
    potential_clip: float = 100.0,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the Lennard-Jones (LJ) body integral over the exterior solvent region.

    Shapes:
    - pred_sdf: [Q] or [B, Q] (predicted SDF fields for query points)
    - query_points: [Q, 3] or [B, Q, 3] (3D coordinates of query probes)
    - coords: [N, 3] or [B, N, 3] (3D coordinates of all atoms)
    - epsilon_lj: [N] or [B, N] (LJ epsilon parameter per atom)
    - sigma_lj: [N] or [B, N] (LJ sigma parameter per atom)
    - atom_mask: [N] or [B, N] (mask for valid atoms in padded batches)
    - mask: [Q] or [B, Q] or None (mask for valid query points)
    - returns: [], a single scalar Tensor representing the averaged loss.
    """
    # Handle single sample inputs by adding a batch dimension
    if pred_sdf.ndim == 1:
        pred_sdf = pred_sdf.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        epsilon_lj = epsilon_lj.unsqueeze(0)
        sigma_lj = sigma_lj.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    # Compute the per-query solvent interaction energy in chunks so large
    # proteins do not allocate a full [B, Q, N] distance matrix at once.
    lj_energy = chunked_lj_potential_sum(
        query_points,
        coords,
        epsilon_lj,
        sigma_lj,
        atom_mask,
        dist_eps=dist_eps,
        potential_clip=potential_clip,
    )
    
    # 5. Extract the exterior solvent region mask using a smooth Heaviside step function.
    # (Inside protein: SDF < 0 -> 0; Outside protein: SDF > 0 -> 1)
    exterior = smooth_heaviside(pred_sdf, eps_h)
    
    # 6. Only integrate the L-J potential over the valid exterior solvent region.
    integrand = lj_energy * exterior
    
    if domain_volume is None:
        if reduction == "none":
            return pred_sdf.new_tensor(float(rho_0)) * _masked_monte_carlo_integral(
                integrand,
                domain_volume=None,
                mask=mask,
                reduction=reduction,
            )
        if mask is not None:
            if not torch.any(mask):
                return pred_sdf.new_zeros(())
            integrand = integrand[mask]
        return pred_sdf.new_tensor(float(rho_0)) * integrand.mean()

    integral = _masked_monte_carlo_integral(
        integrand,
        domain_volume=domain_volume,
        mask=mask,
        reduction=reduction,
    )
    return pred_sdf.new_tensor(float(rho_0)) * integral
