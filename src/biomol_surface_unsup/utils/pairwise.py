from __future__ import annotations

import torch


DEFAULT_QUERY_CHUNK_SIZE = 128


def _chunk_bounds(num_queries: int, chunk_size: int) -> list[tuple[int, int]]:
    size = max(1, int(chunk_size))
    return [(start, min(start + size, num_queries)) for start in range(0, num_queries, size)]


def chunked_atomic_union_sdf(
    coords: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> torch.Tensor:
    """Approximate union-of-spheres SDF using query chunks to cap memory.

    Supports:
    - coords: [N, 3], radii: [N], query_points: [Q, 3] -> [Q]
    - coords: [B, N, 3], radii: [B, N], query_points: [B, Q, 3] -> [B, Q]
    """
    squeeze_batch = query_points.ndim == 2
    if squeeze_batch:
        coords = coords.unsqueeze(0)
        radii = radii.unsqueeze(0)
        query_points = query_points.unsqueeze(0)

    outputs = []
    for start, end in _chunk_bounds(query_points.shape[1], chunk_size):
        query_chunk = query_points[:, start:end]
        pairwise = torch.cdist(query_chunk, coords)
        outputs.append((pairwise - radii.unsqueeze(-2)).amin(dim=-1))

    result = torch.cat(outputs, dim=1)
    return result.squeeze(0) if squeeze_batch else result


def chunked_smooth_atomic_union_field(
    coords: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Smooth union-of-spheres field using query chunks to cap memory."""
    squeeze_batch = query_points.ndim == 2
    if squeeze_batch:
        coords = coords.unsqueeze(0)
        radii = radii.unsqueeze(0)
        query_points = query_points.unsqueeze(0)

    outputs = []
    scale = float(temperature)
    for start, end in _chunk_bounds(query_points.shape[1], chunk_size):
        query_chunk = query_points[:, start:end]
        pairwise = torch.cdist(query_chunk, coords) - radii.unsqueeze(-2) # [B, chunk_size, N]
        outputs.append(-torch.logsumexp(-scale * pairwise, dim=-1) / scale) # list -torch.logsumexp(-scale * pairwise, dim=-1:[B, chunk_size]

    result = torch.cat(outputs, dim=1)
    return result.squeeze(0) if squeeze_batch else result


def chunked_lj_potential_sum(
    query_points: torch.Tensor,
    coords: torch.Tensor,
    epsilon_lj: torch.Tensor,
    sigma_lj: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    dist_eps: float = 1.5,
    potential_clip: float = 100.0,
) -> torch.Tensor:
    """Compute per-query summed LJ energy using query chunks to cap memory."""
    squeeze_batch = query_points.ndim == 2
    if squeeze_batch:
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        epsilon_lj = epsilon_lj.unsqueeze(0)
        sigma_lj = sigma_lj.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)

    outputs = []
    for start, end in _chunk_bounds(query_points.shape[1], chunk_size):
        query_chunk = query_points[:, start:end]
        dists = torch.cdist(query_chunk, coords).clamp_min(float(dist_eps))
        sigma_r6 = (sigma_lj.unsqueeze(1) / dists).pow(6)
        potential = 4.0 * epsilon_lj.unsqueeze(1) * (sigma_r6.pow(2) - sigma_r6)
        potential = potential.clamp(min=-float(potential_clip), max=float(potential_clip))
        outputs.append((potential * atom_mask.unsqueeze(1).to(potential.dtype)).sum(dim=-1))

    result = torch.cat(outputs, dim=1)
    return result.squeeze(0) if squeeze_batch else result


def chunked_coulomb_field_squared_sum(
    query_points: torch.Tensor,
    coords: torch.Tensor,
    charges: torch.Tensor,
    atom_mask: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
    dist_eps: float = 1.0,
) -> torch.Tensor:
    """Compute |sum_i q_i (x-x_i) / |x-x_i|^3|^2 per query using query chunks."""
    squeeze_batch = query_points.ndim == 2
    if squeeze_batch:
        query_points = query_points.unsqueeze(0)
        coords = coords.unsqueeze(0)
        charges = charges.unsqueeze(0)
        atom_mask = atom_mask.unsqueeze(0)

    outputs = []
    for start, end in _chunk_bounds(query_points.shape[1], chunk_size):
        query_chunk = query_points[:, start:end]
        disp = query_chunk.unsqueeze(2) - coords.unsqueeze(1)
        dists = disp.norm(dim=-1).clamp_min(float(dist_eps))
        inv_r3 = dists.reciprocal().pow(3)
        weighted = disp * (charges.unsqueeze(1) * atom_mask.unsqueeze(1).to(disp.dtype) * inv_r3).unsqueeze(-1)
        field = weighted.sum(dim=2)
        outputs.append(field.pow(2).sum(dim=-1))

    result = torch.cat(outputs, dim=1)
    return result.squeeze(0) if squeeze_batch else result
