from __future__ import annotations

import torch

from .area import _masked_monte_carlo_integral
from .heaviside import smooth_heaviside


def pressure_volume_loss(
    pred_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
    pressure: float = 0.01,
    eps: float = 0.1,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pressure-volume term.

    When ``domain_volume`` is provided, interpret the integral physically over the
    solute cavity (``phi < 0``). When it is omitted, preserve the older
    mean-over-samples behavior used by the unit tests and earlier training code,
    which treated this term as an exterior fraction surrogate.
    """
    if domain_volume is None:
        exterior = smooth_heaviside(pred_sdf, eps)
        if reduction == "none":
            return pred_sdf.new_tensor(float(pressure)) * _masked_monte_carlo_integral(
                exterior,
                domain_volume=None,
                mask=mask,
                reduction=reduction,
            )
        if mask is not None:
            if not torch.any(mask):
                return pred_sdf.new_zeros(())
            exterior = exterior[mask]
        return pred_sdf.new_tensor(float(pressure)) * exterior.mean()

    # Physical Monte Carlo path: SDF < 0 inside the solute cavity.
    interior = smooth_heaviside(-pred_sdf, eps)
    integral = _masked_monte_carlo_integral(
        interior,
        domain_volume=domain_volume,
        mask=mask,
        reduction=reduction,
    )
    return pred_sdf.new_tensor(float(pressure)) * integral
