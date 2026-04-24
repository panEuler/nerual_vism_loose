import torch

def random_rigid_transform(coords: torch.Tensor) -> torch.Tensor:
    # TODO: add optional SE(3) augmentation
    return coords