import torch
import torch.nn as nn
from .atom_features import AtomFeatureEmbedding


class GlobalFeatureEncoder(nn.Module):
    def __init__(self, num_atom_types: int, atom_embed_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.atom_embedding = AtomFeatureEmbedding(num_atom_types, atom_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(3 + atom_embed_dim + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, coords, atom_types, radii, charges=None, epsilon=None, sigma=None, atom_mask=None):
        squeeze_batch = coords.ndim == 2
        if squeeze_batch:
            coords = coords.unsqueeze(0)
            atom_types = atom_types.unsqueeze(0)
            radii = radii.unsqueeze(0)
            charges = None if charges is None else charges.unsqueeze(0)
            epsilon = None if epsilon is None else epsilon.unsqueeze(0)
            sigma = None if sigma is None else sigma.unsqueeze(0)
            atom_mask = None if atom_mask is None else atom_mask.unsqueeze(0)

        if charges is None:
            charges = torch.zeros_like(radii)
        if epsilon is None:
            epsilon = torch.zeros_like(radii)
        if sigma is None:
            sigma = torch.zeros_like(radii)

        if atom_mask is None:
            atom_mask = torch.ones(atom_types.shape, dtype=torch.bool, device=atom_types.device)

        atom_emb = self.atom_embedding(atom_types)
        # Use center-of-mass-relative coordinates for translation invariance
        mask_float = atom_mask.unsqueeze(-1).float()  # [B, N, 1]
        com = (coords * mask_float).sum(dim=1, keepdim=True) / mask_float.sum(dim=1, keepdim=True).clamp_min(1)  # [B, 1, 3]
        rel_coords = coords - com  # [B, N, 3]
        x = torch.cat(
            [
                rel_coords,
                atom_emb,
                radii.unsqueeze(-1),
                charges.unsqueeze(-1),
                epsilon.unsqueeze(-1),
                sigma.unsqueeze(-1),
            ],
            dim=-1,
        )  # [B, N, F]
        h = self.mlp(x) * atom_mask.unsqueeze(-1).to(x.dtype)
        denom = atom_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        pooled = h.sum(dim=1) / denom
        out = self.out(pooled)
        if squeeze_batch:
            return out.squeeze(0)
        return out
