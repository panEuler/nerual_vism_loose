import torch
import torch.nn as nn

class AtomFeatureEmbedding(nn.Module):
    def __init__(self, num_atom_types: int, atom_embed_dim: int):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, atom_embed_dim)

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        return self.atom_embedding(atom_types)