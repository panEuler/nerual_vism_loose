from __future__ import annotations

import math

import torch
import torch.nn as nn


class SirenLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.omega_0 = float(omega_0)
        self.is_first = bool(is_first)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3, omega_0: float = 30.0) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("SirenDecoder requires num_layers >= 2")
        layers: list[nn.Module] = [SirenLayer(in_dim, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 2):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_dim) / omega_0
            self.out.weight.uniform_(-bound, bound)
            self.out.bias.uniform_(-bound, bound)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(z)).squeeze(-1)
