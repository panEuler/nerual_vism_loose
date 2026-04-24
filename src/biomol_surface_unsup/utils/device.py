import torch

def get_device(name: str):
    if name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"