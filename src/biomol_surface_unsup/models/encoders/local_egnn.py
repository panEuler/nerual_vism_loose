import torch.nn as nn

class LocalEGNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        # TODO: replace with real EGNN block
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, features, mask):
        # TODO: use coordinates explicitly in message passing
        h = self.net(features) * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        return h.sum(dim=1) / denom