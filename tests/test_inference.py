from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.inference import load_processed_molecule, predict_sdf
from biomol_surface_unsup.geometry.sdf_ops import box_sdf
from biomol_surface_unsup.models.surface_model import SurfaceModel
from tests.test_dataset import _write_processed_sample


def test_predict_sdf_matches_non_chunked_forward(tmp_path: Path) -> None:
    torch.manual_seed(0)
    sample_dir = tmp_path / "1ABC_TEST"
    _write_processed_sample(sample_dir, "1ABC_TEST_A", num_atoms=4)
    molecule = load_processed_molecule(sample_dir)
    query_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, -0.3],
            [2.0, -0.1, 0.4],
            [3.0, 0.3, 0.1],
            [1.5, -0.2, -0.2],
        ],
        dtype=torch.float32,
    )

    model = SurfaceModel(num_atom_types=16)
    with torch.no_grad():
        expected = model(
            molecule["coords"],
            molecule["atom_types"],
            molecule["radii"],
            query_points,
            charges=molecule["charges"],
            epsilon=molecule["epsilon"],
            sigma=molecule["sigma"],
        )["sdf"]

    actual = predict_sdf(
        model=model,
        molecule=molecule,
        query_points=query_points,
        chunk_size=2,
        device="cpu",
    )

    assert tuple(actual.shape) == (5,)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_predict_sdf_passes_box_base_bounds(tmp_path: Path) -> None:
    sample_dir = tmp_path / "1ABC_TEST"
    _write_processed_sample(sample_dir, "1ABC_TEST_A", num_atoms=4)
    molecule = load_processed_molecule(sample_dir)
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    bbox_lower = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float32)
    bbox_upper = torch.tensor([5.0, 2.0, 2.0], dtype=torch.float32)
    model = SurfaceModel(num_atom_types=16, sdf_base_type="box", zero_init_output=True)

    actual = predict_sdf(
        model=model,
        molecule=molecule,
        query_points=query_points,
        chunk_size=2,
        device="cpu",
        bbox_lower=bbox_lower,
        bbox_upper=bbox_upper,
    )
    expected = box_sdf(query_points, bbox_lower, bbox_upper)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_load_processed_molecule_returns_physical_atom_features(tmp_path: Path) -> None:
    sample_dir = tmp_path / "1ABC_TEST"
    _write_processed_sample(sample_dir, "1ABC_TEST_A", num_atoms=4)

    molecule = load_processed_molecule(sample_dir)

    assert tuple(molecule["coords"].shape) == (4, 3)
    assert tuple(molecule["atom_types"].shape) == (4,)
    assert tuple(molecule["radii"].shape) == (4,)
    assert tuple(molecule["charges"].shape) == (4,)
    assert tuple(molecule["epsilon"].shape) == (4,)
    assert tuple(molecule["sigma"].shape) == (4,)
    assert np.isfinite(molecule["charges"].numpy()).all()
