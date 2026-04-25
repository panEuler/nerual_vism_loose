from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None

from biomol_surface_unsup.utils.pairwise import chunked_atomic_union_sdf


QUERY_GROUP_GLOBAL = 0
QUERY_GROUP_CONTAINMENT = 1
QUERY_GROUP_SURFACE_BAND = 2
QUERY_GROUP_AREA = 3


def _infer_bond_pairs(
    coords: torch.Tensor,
    radii: torch.Tensor,
    max_neighbors: int = 8,
    chunk_size: int = 256,
) -> torch.Tensor:
    if coords.shape[0] < 2:
        return torch.empty((0, 2), dtype=torch.long, device=coords.device)

    neighbor_k = min(max(1, int(max_neighbors)), coords.shape[0] - 1)
    pair_chunks: list[torch.Tensor] = []

    for start in range(0, coords.shape[0], max(1, int(chunk_size))):
        end = min(start + max(1, int(chunk_size)), coords.shape[0])
        coord_chunk = coords[start:end]
        dist_chunk = torch.cdist(coord_chunk, coords)

        row_indices = torch.arange(start, end, device=coords.device)
        dist_chunk[torch.arange(end - start, device=coords.device), row_indices] = float("inf")

        cutoff_chunk = radii[start:end].unsqueeze(1) + radii.unsqueeze(0) + 0.5
        dist_chunk = dist_chunk.masked_fill(dist_chunk > cutoff_chunk, float("inf"))

        chunk_dists, chunk_indices = torch.topk(dist_chunk, k=neighbor_k, dim=-1, largest=False)
        valid = torch.isfinite(chunk_dists)
        if not torch.any(valid):
            continue

        chunk_rows = row_indices.unsqueeze(1).expand_as(chunk_indices)
        pair_rows = chunk_rows[valid]
        pair_cols = chunk_indices[valid]
        pair_first = torch.minimum(pair_rows, pair_cols)
        pair_second = torch.maximum(pair_rows, pair_cols)
        pair_chunks.append(torch.stack([pair_first, pair_second], dim=1))

    if not pair_chunks:
        return torch.empty((0, 2), dtype=torch.long, device=coords.device)

    return torch.unique(torch.cat(pair_chunks, dim=0), dim=0)


def _sample_convex_hull_interior(
    coords: torch.Tensor,
    radii: torch.Tensor,
    num_points: int,
    max_attempt_factor: int = 8,
) -> torch.Tensor:
    if num_points <= 0:
        return coords.new_empty((0, 3))

    accepted: list[torch.Tensor] = []
    attempts = 0
    max_attempts = max(num_points * max_attempt_factor, num_points)
    while sum(point.shape[0] for point in accepted) < num_points and attempts < max_attempts:
        remaining = num_points - sum(point.shape[0] for point in accepted)
        weights = torch.rand(remaining, coords.shape[0], dtype=coords.dtype, device=coords.device)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        candidates = weights @ coords
        candidate_sdf = approximate_atomic_union_sdf(coords, radii, candidates)
        inside = candidates[candidate_sdf <= 0.0]
        if inside.numel() > 0:
            accepted.append(inside[:remaining])
        attempts += remaining

    if accepted:
        return torch.cat(accepted, dim=0)[:num_points]
    return coords.new_empty((0, 3))


def _compute_bbox(coords: torch.Tensor, radii: torch.Tensor | None, padding: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bbox bounds with shape [3]."""
    if radii is not None:
        lower = (coords - radii.unsqueeze(-1)).amin(dim=0) - padding
        upper = (coords + radii.unsqueeze(-1)).amax(dim=0) + padding
    else:
        lower = coords.amin(dim=0) - padding
        upper = coords.amax(dim=0) + padding
    return lower, upper


def _sample_box_surface_band(
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_points: int,
    band_width: float,
) -> torch.Tensor:
    """Sample points in a thin band around the six faces of an axis-aligned box."""
    if num_points <= 0:
        return lower.new_empty((0, 3))

    points = lower.unsqueeze(0) + torch.rand(
        num_points,
        3,
        dtype=lower.dtype,
        device=lower.device,
    ) * (upper - lower).unsqueeze(0)

    axes = torch.randint(0, 3, (num_points,), device=lower.device)
    use_upper_face = torch.randint(0, 2, (num_points,), device=lower.device, dtype=torch.bool)
    face_values = torch.where(use_upper_face, upper[axes], lower[axes])
    jitter = (2.0 * torch.rand(num_points, dtype=lower.dtype, device=lower.device) - 1.0) * float(band_width)
    points[torch.arange(num_points, device=lower.device), axes] = face_values + jitter
    return points


def _sample_uniform_box(lower: torch.Tensor, upper: torch.Tensor, num_points: int) -> torch.Tensor:
    if num_points <= 0:
        return lower.new_empty((0, 3))
    return lower.unsqueeze(0) + torch.rand(
        num_points,
        3,
        dtype=lower.dtype,
        device=lower.device,
    ) * (upper - lower).unsqueeze(0)


def approximate_atomic_union_sdf(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    """Toy atomic-union SDF approximation with shape [Q].

    TODO: this is still a toy proxy for the molecular surface. It approximates the
    union-of-spheres field using per-atom Euclidean distance minus atom radius.
    """
    return chunked_atomic_union_sdf(coords, radii, query_points)


def sample_query_points(
    coords: Any,
    num_query_points: int,
    padding: float,
    radii: Any | None = None,
    containment_jitter: float = 0.15,
    surface_band_width: float = 0.25,
    num_area_points: int = 0,
    initialization_mode: str = "tight_atomic",
    loose_surface_padding: float | None = None,
    domain_padding: float | None = None,
) -> dict[str, Any]:
    """Create hierarchical toy query samples.

    Shapes for torch path:
    - coords: [N, 3]
    - radii: [N]
    - query_points: [Q, 3]
    - query_group: [Q]
    - containment_points: [C, 3]

    Sampling groups:
    - global: bbox-uniform samples
    - containment: atom-centered / near-atom anchors
    - surface-band: samples near a toy atomic-union narrow band
    - area: extra bbox-uniform samples used only to reduce area integral variance
    """
    if torch is None or not isinstance(coords, torch.Tensor):
        raise RuntimeError("sample_query_points requires torch in the current toy training path")

    if num_query_points <= 0:
        raise ValueError("num_query_points must be positive")
    if radii is None:
        radii = torch.full((coords.shape[0],), 1.0, dtype=coords.dtype, device=coords.device)

    num_global = max(1, num_query_points // 2)
    num_containment = max(1, num_query_points // 4)
    num_surface = max(1, num_query_points - num_global - num_containment)
    num_area = max(0, int(num_area_points))
    total = num_global + num_containment + num_surface
    if total != num_query_points:
        num_surface += num_query_points - total

    mode = str(initialization_mode).lower()
    if mode not in {"tight_atomic", "loose_box"}:
        raise ValueError("initialization_mode must be either 'tight_atomic' or 'loose_box'")

    loose_surface_padding = float(padding if loose_surface_padding is None else loose_surface_padding)
    domain_padding = float(padding if domain_padding is None else domain_padding)
    if mode == "loose_box" and domain_padding <= loose_surface_padding:
        raise ValueError("domain_padding must be greater than loose_surface_padding for loose_box initialization")

    if mode == "loose_box":
        surface_lower, surface_upper = _compute_bbox(coords, radii, loose_surface_padding)
        domain_lower, domain_upper = _compute_bbox(coords, radii, domain_padding)
    else:
        domain_lower, domain_upper = _compute_bbox(coords, radii, padding)
        surface_lower, surface_upper = domain_lower, domain_upper

    # [Qg, 3]
    global_samples = _sample_uniform_box(domain_lower, domain_upper, num_global)

    num_atom_containment = max(1, num_containment // 2)
    remaining_containment = num_containment - num_atom_containment
    num_midpoint = remaining_containment // 2
    num_hull = remaining_containment - num_midpoint

    atom_index = torch.arange(num_atom_containment, device=coords.device) % coords.shape[0]
    base_centers = coords[atom_index]
    jitter_dir = torch.randn(num_atom_containment, 3, dtype=coords.dtype, device=coords.device)
    jitter_dir = jitter_dir / jitter_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    jitter_scale = (radii[atom_index] * containment_jitter).unsqueeze(-1)
    atom_containment = base_centers + jitter_dir * jitter_scale

    bond_pairs = _infer_bond_pairs(coords, radii)
    if bond_pairs.shape[0] > 0 and num_midpoint > 0:
        midpoint_index = torch.arange(num_midpoint, device=coords.device) % bond_pairs.shape[0]
        selected_pairs = bond_pairs[midpoint_index]
        midpoint_points = 0.5 * (coords[selected_pairs[:, 0]] + coords[selected_pairs[:, 1]])
    else:
        midpoint_points = coords.new_empty((0, 3))

    hull_points = _sample_convex_hull_interior(coords, radii, num_hull)
    containment_chunks = [atom_containment, midpoint_points, hull_points]
    containment_points = torch.cat([chunk for chunk in containment_chunks if chunk.shape[0] > 0], dim=0)
    if containment_points.shape[0] < num_containment:
        pad_count = num_containment - containment_points.shape[0]
        fallback_index = torch.arange(pad_count, device=coords.device) % coords.shape[0]
        fallback_dir = torch.randn(pad_count, 3, dtype=coords.dtype, device=coords.device)
        fallback_dir = fallback_dir / fallback_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        fallback_scale = (radii[fallback_index] * 1e-3).unsqueeze(-1)
        fallback_points = coords[fallback_index] + fallback_dir * fallback_scale
        containment_points = torch.cat([containment_points, fallback_points], dim=0)
    containment_points = containment_points[:num_containment]

    if mode == "loose_box":
        surface_samples = _sample_box_surface_band(
            surface_lower,
            surface_upper,
            num_surface,
            band_width=surface_band_width,
        )
    else:
        candidate_count = max(num_surface * 8, num_surface)
        candidate_points = domain_lower.unsqueeze(0) + torch.rand(
            candidate_count,
            3,
            dtype=coords.dtype,
            device=coords.device,
        ) * (domain_upper - domain_lower).unsqueeze(0)  # [K, 3]
        candidate_field = approximate_atomic_union_sdf(coords, radii, candidate_points)  # [K]
        near_surface_mask = candidate_field.abs() <= surface_band_width
        if int(near_surface_mask.sum().item()) >= num_surface:
            surface_samples = candidate_points[near_surface_mask][:num_surface]  # [Qs, 3]
        else:
            sort_index = candidate_field.abs().argsort()
            surface_samples = candidate_points[sort_index[:num_surface]]  # [Qs, 3]

    area_samples = _sample_uniform_box(domain_lower, domain_upper, num_area)

    point_chunks = [global_samples, containment_points, surface_samples]
    group_chunks = [
        torch.full((num_global,), QUERY_GROUP_GLOBAL, dtype=torch.long, device=coords.device),
        torch.full((num_containment,), QUERY_GROUP_CONTAINMENT, dtype=torch.long, device=coords.device),
        torch.full((num_surface,), QUERY_GROUP_SURFACE_BAND, dtype=torch.long, device=coords.device),
    ]
    if num_area > 0:
        point_chunks.append(area_samples)
        group_chunks.append(torch.full((num_area,), QUERY_GROUP_AREA, dtype=torch.long, device=coords.device))

    query_points = torch.cat(point_chunks, dim=0)  # [Q, 3]
    query_group = torch.cat(group_chunks, dim=0)  # [Q]

    sampling_counts = {
        "global": int(num_global),
        "containment": int(num_containment),
        "surface_band": int(num_surface),
    }
    if num_area > 0:
        sampling_counts["area"] = int(num_area)

    return {
        "query_points": query_points,
        "query_group": query_group,
        "containment_points": containment_points,
        "surface_bbox_lower": surface_lower,
        "surface_bbox_upper": surface_upper,
        "domain_bbox_lower": domain_lower,
        "domain_bbox_upper": domain_upper,
        "bbox_lower": domain_lower,
        "bbox_upper": domain_upper,
        "bbox_volume": (domain_upper - domain_lower).prod(),
        "sampling_counts": sampling_counts,
    }


def sample_surface_band_points(coords: Any, num_points: int, radii: Any, padding: float = 0.0):
    """Return only the toy near-surface-band samples with shape [Q, 3]."""
    sampling = sample_query_points(
        coords=coords,
        num_query_points=num_points,
        padding=padding,
        radii=radii,
    )
    mask = sampling["query_group"] == QUERY_GROUP_SURFACE_BAND
    return sampling["query_points"][mask]
