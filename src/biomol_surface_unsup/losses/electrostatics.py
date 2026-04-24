from __future__ import annotations

import math

import torch

from biomol_surface_unsup.utils.pairwise import chunked_coulomb_field_squared_sum

from .area import _masked_monte_carlo_integral
from .heaviside import smooth_heaviside


COULOMB_CONSTANT_KJ_MOL_ANGSTROM_E2 = 1389.35457644382


def electrostatic_free_energy_cfa(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    coords: torch.Tensor,
    charges: torch.Tensor,
    atom_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps_solvent: float = 78.0,
    eps_solute: float = 1.0,
    eps_h: float = 0.1,
    dist_eps: float = 1.0,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Coulomb-field approximation to VISM electrostatic free energy.

    G_elec ≈ (k_e / 8π) * (1/eps_solvent - 1/eps_solute) *
             ∫_{Ω_w} | Σ_i q_i (x - x_i) / |x - x_i|^3 |^2 dV

    Coordinates are assumed to be in Angstrom and charges in elementary charge.
    """
    if pred_sdf.ndim == 1:
        pred_sdf = pred_sdf.unsqueeze(0)
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        charges = charges.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be either 'mean' or 'none'")
    if abs(float(eps_solvent) - float(eps_solute)) < 1e-12:
        if reduction == "none":
            return pred_sdf.new_zeros((pred_sdf.shape[0],))
        return pred_sdf.new_zeros(())

    with torch.no_grad():
        field_sq = chunked_coulomb_field_squared_sum(
            query_points,
            coords,
            charges,
            atom_mask,
            dist_eps=dist_eps,
        )

    exterior = smooth_heaviside(pred_sdf, eps_h)
    integrand = field_sq * exterior

    if domain_volume is None:
        if reduction == "none":
            integral = _masked_monte_carlo_integral(
                integrand,
                domain_volume=None,
                mask=mask,
                reduction=reduction,
            )
        else:
            if mask is not None:
                if not torch.any(mask):
                    return pred_sdf.new_zeros(())
                integrand = integrand[mask]
            integral = integrand.mean()
    else:
        integral = _masked_monte_carlo_integral(
            integrand,
            domain_volume=domain_volume,
            mask=mask,
            reduction=reduction,
        )

    coefficient = (
        COULOMB_CONSTANT_KJ_MOL_ANGSTROM_E2
        / (8.0 * math.pi)
        * (1.0 / float(eps_solvent) - 1.0 / float(eps_solute))
    )
    return pred_sdf.new_tensor(coefficient) * integral
