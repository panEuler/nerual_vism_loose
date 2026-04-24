import torch

def concat_fusion(z_local: torch.Tensor, z_global: torch.Tensor) -> torch.Tensor:
    return torch.cat([z_local, z_global], dim=-1)