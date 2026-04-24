from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.training.checkpoint import _checkpoint_model, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"

    original_weight = model.weight.detach().clone()
    saved = save_checkpoint(path, model, optimizer=optimizer, epoch=3, step=42, metrics={"loss": 1.23})
    assert saved.exists()

    with torch.no_grad():
        model.weight.zero_()

    checkpoint = load_checkpoint(path, model, optimizer=optimizer)
    assert checkpoint["epoch"] == 3
    assert checkpoint["step"] == 42
    assert checkpoint["metrics"]["loss"] == pytest.approx(1.23)
    assert torch.allclose(model.weight, original_weight)


def test_checkpoint_helpers_use_wrapped_module_when_present(tmp_path: Path) -> None:
    class WrappedModel:
        def __init__(self):
            self.module = torch.nn.Linear(4, 2)

    wrapped = WrappedModel()
    optimizer = torch.optim.AdamW(wrapped.module.parameters(), lr=1e-3)
    path = tmp_path / "wrapped.pt"

    original_weight = wrapped.module.weight.detach().clone()
    save_checkpoint(path, wrapped, optimizer=optimizer, epoch=1, step=2)

    with torch.no_grad():
        wrapped.module.weight.zero_()

    load_checkpoint(path, wrapped, optimizer=optimizer)
    assert _checkpoint_model(wrapped) is wrapped.module
    assert torch.allclose(wrapped.module.weight, original_weight)
