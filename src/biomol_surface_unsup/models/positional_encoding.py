from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierEncoder(nn.Module):
    def __init__(self, d_in: int = 3, n_freq: int = 6, include_input: bool = True) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.n_freq = int(n_freq)
        self.include_input = bool(include_input)
        self.register_buffer("freq_bands", 2.0 ** torch.arange(self.n_freq, dtype=torch.float32), persistent=False)
        self.out_dim = self.d_in * (2 * self.n_freq + (1 if self.include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = [x] if self.include_input else []
        for freq in self.freq_bands.to(device=x.device, dtype=x.dtype):
            scaled = x * (2.0 * math.pi * freq)
            encodings.append(torch.sin(scaled))
            encodings.append(torch.cos(scaled))
        if not encodings:
            return x.new_empty((*x.shape[:-1], 0))
        return torch.cat(encodings, dim=-1)
