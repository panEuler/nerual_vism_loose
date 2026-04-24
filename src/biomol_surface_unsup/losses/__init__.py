"""Public loss API.

The project currently exposes both the modern batched builder (`build_loss_fn`)
and a tiny legacy helper (`build_loss`) kept for older tests and scripts.
Similarly, `lj_body_integral` is re-exported here from its canonical VDW module.
"""

from biomol_surface_unsup.losses.loss_builder import build_loss, build_loss_fn
from biomol_surface_unsup.losses.pressure_volume import pressure_volume_loss
from biomol_surface_unsup.losses.vdw import lj_body_integral

__all__ = [
    "build_loss",
    "build_loss_fn",
    "pressure_volume_loss",
    "lj_body_integral",
]
