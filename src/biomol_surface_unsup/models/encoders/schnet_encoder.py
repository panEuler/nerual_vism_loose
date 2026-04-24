from __future__ import annotations

import torch
import torch.nn as nn


class ContinuousFilterBlock(nn.Module):
    def __init__(self, hidden_dim: int, rbf_dim: int) -> None:
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, rbf: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        filt = self.filter_net(rbf)
        messages = h * filt * mask.unsqueeze(-1).to(h.dtype)
        pooled = messages.sum(dim=2, keepdim=True)
        denom = mask.sum(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1).to(h.dtype)
        pooled = (pooled / denom).expand_as(h)
        update = self.update(torch.cat([h, pooled], dim=-1))
        return (h + update) * mask.unsqueeze(-1).to(h.dtype)


class SchNetEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, rbf_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.rbf_dim = int(rbf_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [ContinuousFilterBlock(hidden_dim=hidden_dim, rbf_dim=self.rbf_dim) for _ in range(int(num_layers))]
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squeeze_batch = features.ndim == 3
        if squeeze_batch:
            features = features.unsqueeze(0)
            mask = mask.unsqueeze(0)

        rbf = features[..., -(self.rbf_dim + 1) : -1]
        h = self.input_proj(features) * mask.unsqueeze(-1).to(features.dtype)
        for block in self.blocks:
            h = block(h, rbf, mask)

        denom = mask.sum(dim=2, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=2) / denom
        out = self.out(pooled)
        if squeeze_batch:
            return out.squeeze(0)
        return out
