from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.geometry.sdf_ops import box_sdf


def test_box_sdf_uses_negative_inside_positive_outside_convention() -> None:
    lower = torch.tensor([-1.0, -2.0, -3.0], dtype=torch.float32)
    upper = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    query_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    phi = box_sdf(query_points, lower, upper)

    assert phi[0].item() < 0.0
    assert phi[1].item() == pytest.approx(0.0, abs=1e-6)
    assert phi[2].item() > 0.0


def test_box_sdf_supports_batched_bounds() -> None:
    lower = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]], dtype=torch.float32)
    upper = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    query_points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.25, 0.0, 0.0]],
            [[1.9, 0.0, 0.0], [2.5, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    phi = box_sdf(query_points, lower, upper)

    assert tuple(phi.shape) == (2, 2)
    assert phi[0, 0].item() < 0.0
    assert phi[0, 1].item() > 0.0
    assert phi[1, 0].item() < 0.0
    assert phi[1, 1].item() > 0.0
