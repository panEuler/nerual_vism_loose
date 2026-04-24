import torch.nn as nn


class SDFDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(int(num_layers) - 2, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z).squeeze(-1)
