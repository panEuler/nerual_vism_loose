import torch
import torch.nn as nn


class LocalDeepSetsEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeeze_batch = features.ndim == 3
        if squeeze_batch:
            features = features.unsqueeze(0)
            mask = mask.unsqueeze(0)

        h = self.phi(features) * mask.unsqueeze(-1).to(features.dtype)
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=2) / denom
        out = self.rho(pooled)
        if squeeze_batch:
            return out.squeeze(0)
        return out
