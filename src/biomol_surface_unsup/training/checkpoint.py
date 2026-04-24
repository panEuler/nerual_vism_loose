from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _checkpoint_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    epoch: int = 0,
    step: int = 0,
    metrics: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": _checkpoint_model(model).state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, model, optimizer=None, map_location: str = "cpu"):
    ckpt = torch.load(path, map_location=map_location)
    _checkpoint_model(model).load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
