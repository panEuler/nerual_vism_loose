import torch

def radius_knn(query_points: torch.Tensor, atom_coords: torch.Tensor, cutoff: float, max_neighbors: int):
    dmat = torch.cdist(query_points, atom_coords)
    within = dmat <= cutoff
    masked = dmat.clone()
    masked[~within] = float("inf")
    k = min(max_neighbors, atom_coords.shape[0])
    dist, idx = torch.topk(masked, k=k, dim=1, largest=False)
    valid = torch.isfinite(dist)
    idx = idx.masked_fill(~valid, -1)
    dist = dist.masked_fill(~valid, 0.0)
    return idx, dist, valid