"""Encoder modules."""

from biomol_surface_unsup.models.encoders.local_deepsets import LocalDeepSetsEncoder
from biomol_surface_unsup.models.encoders.schnet_encoder import SchNetEncoder

__all__ = ["LocalDeepSetsEncoder", "SchNetEncoder"]
