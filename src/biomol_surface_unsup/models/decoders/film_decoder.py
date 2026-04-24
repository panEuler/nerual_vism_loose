import torch
import torch.nn as nn

class FiLMDecoder(nn.Module):
    def __init__(self, local_dim: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Linear(global_dim, local_dim)
        self.beta = nn.Linear(global_dim, local_dim)
        self.mlp = nn.Sequential(
            nn.Linear(local_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_local: torch.Tensor, z_global: torch.Tensor):
        gamma = self.gamma(z_global)
        beta = self.beta(z_global)
        z = gamma * z_local + beta
        return self.mlp(z).squeeze(-1)