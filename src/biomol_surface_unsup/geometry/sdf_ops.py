import torch

from biomol_surface_unsup.utils.pairwise import chunked_smooth_atomic_union_field


def _align_box_bound(bound: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    bound = torch.as_tensor(bound, dtype=query_points.dtype, device=query_points.device)
    if query_points.ndim == 3 and bound.ndim == 2:
        return bound.unsqueeze(1)
    return bound


def box_sdf(query_points: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Signed distance to an axis-aligned box.

    Negative values are inside the box, zero values are on the box surface, and
    positive values are outside. Supports unbatched ``[Q, 3]`` queries and
    batched ``[B, Q, 3]`` queries with either shared ``[3]`` bounds or per-sample
    ``[B, 3]`` bounds.
    """
    if query_points.shape[-1] != 3:
        raise ValueError(f"query_points must have last dimension 3, got {tuple(query_points.shape)}")

    lower = _align_box_bound(lower, query_points)
    upper = _align_box_bound(upper, query_points)
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    q = (query_points - center).abs() - half
    outside = torch.clamp(q, min=0.0).norm(dim=-1)
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)
    return outside + inside


def sphere_sdf(query_points: torch.Tensor, center: torch.Tensor, radius: torch.Tensor):
    return (query_points - center).norm(dim=-1) - radius


def smooth_min(x: torch.Tensor, dim: int = -1, temperature: float = 10.0):
    return -torch.logsumexp(-temperature * x, dim=dim) / temperature


def atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor):
    return chunked_smooth_atomic_union_field(coords, radii, query_points)
