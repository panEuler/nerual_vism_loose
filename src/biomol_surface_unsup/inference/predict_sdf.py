from __future__ import annotations

from pathlib import Path

import torch

from biomol_surface_unsup.datasets.molecule_dataset import (
    _encode_atom_types,
    _find_sample_prefix,
    _load_npy,
    _required_sample_fields,
)
from biomol_surface_unsup.datasets.sampling import _compute_bbox


def load_processed_molecule(sample_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load one preprocessed protein sample for point-wise SDF inference."""
    sample_dir = Path(sample_dir)
    prefix = _find_sample_prefix(sample_dir)
    arrays = {
        name: _load_npy(sample_dir / filename)
        for name, filename in _required_sample_fields(prefix).items()
    }
    return {
        "coords": torch.as_tensor(arrays["coords"], dtype=torch.float32),
        "atom_types": torch.as_tensor(_encode_atom_types(arrays["atom_types"]), dtype=torch.long),
        "radii": torch.as_tensor(arrays["radii"], dtype=torch.float32),
        "charges": torch.as_tensor(arrays["charges"], dtype=torch.float32),
        "epsilon": torch.as_tensor(arrays["epsilon"], dtype=torch.float32),
        "sigma": torch.as_tensor(arrays["sigma"], dtype=torch.float32),
    }


def _validate_query_points(query_points: torch.Tensor) -> torch.Tensor:
    if query_points.ndim != 2 or query_points.shape[-1] != 3:
        raise ValueError(f"query_points must have shape [Q, 3], got {tuple(query_points.shape)}")
    return query_points


@torch.no_grad()
def predict_sdf(
    model: torch.nn.Module,
    molecule: dict[str, torch.Tensor],
    query_points: torch.Tensor,
    device: str | torch.device = "cpu",
    chunk_size: int = 8192,
    bbox_lower: torch.Tensor | None = None,
    bbox_upper: torch.Tensor | None = None,
    loose_surface_padding: float | None = None,
) -> torch.Tensor:
    """Predict SDF values for arbitrary xyz query points on one molecule."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    query_points = _validate_query_points(torch.as_tensor(query_points, dtype=torch.float32))
    model = model.to(device)
    model.eval()

    coords = molecule["coords"].to(device)
    atom_types = molecule["atom_types"].to(device)
    radii = molecule["radii"].to(device)
    charges = molecule["charges"].to(device)
    epsilon = molecule["epsilon"].to(device)
    sigma = molecule["sigma"].to(device)
    model_uses_box_base = str(getattr(model, "sdf_base_type", "none")).lower() == "box"
    if model_uses_box_base:
        if bbox_lower is None:
            bbox_lower = molecule.get("surface_bbox_lower")
        if bbox_upper is None:
            bbox_upper = molecule.get("surface_bbox_upper")
        if bbox_lower is None or bbox_upper is None:
            if loose_surface_padding is None:
                raise ValueError(
                    "loose_surface_padding or explicit bbox_lower/bbox_upper is required "
                    "when predicting with sdf_base.type='box'"
                )
            bbox_lower, bbox_upper = _compute_bbox(coords, radii, float(loose_surface_padding))
        bbox_lower = torch.as_tensor(bbox_lower, dtype=torch.float32, device=device)
        bbox_upper = torch.as_tensor(bbox_upper, dtype=torch.float32, device=device)

    sdf_chunks = []
    for start in range(0, query_points.shape[0], chunk_size):
        query_chunk = query_points[start : start + chunk_size].to(device)
        bbox_kwargs = {}
        if model_uses_box_base:
            bbox_kwargs = {"bbox_lower": bbox_lower, "bbox_upper": bbox_upper}
        out = model(
            coords,
            atom_types,
            radii,
            query_chunk,
            charges=charges,
            epsilon=epsilon,
            sigma=sigma,
            **bbox_kwargs,
        )
        sdf_chunks.append(out["sdf"].detach().cpu())

    if not sdf_chunks:
        return query_points.new_zeros((0,))
    return torch.cat(sdf_chunks, dim=0)
