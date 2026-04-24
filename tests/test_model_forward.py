from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.features.global_features import GlobalFeatureEncoder
from biomol_surface_unsup.geometry.sdf_ops import box_sdf
from biomol_surface_unsup.models.positional_encoding import FourierEncoder
from biomol_surface_unsup.models.surface_model import SurfaceModel


def test_model_forward_single_sample_keeps_compatibility():
    dataset = MoleculeDataset(num_query_points=32)
    sample = dataset[0]
    expected_neighbors = min(sample["coords"].shape[0], 64)

    model = SurfaceModel(num_atom_types=dataset.num_atom_types)
    out = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
    )
    assert out["sdf"].shape == (32,)
    assert out["features"].shape[:2] == (32, expected_neighbors)
    assert out["mask"].shape[:2] == (32, expected_neighbors)


def test_model_forward_batched_uses_atom_and_query_masks():
    dataset = MoleculeDataset(num_samples=2, num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])
    expected_neighbors = min(batch["coords"].shape[1], 64)

    model = SurfaceModel(num_atom_types=dataset.num_atom_types)
    out = model(
        batch["coords"],
        batch["atom_types"],
        batch["radii"],
        batch["query_points"],
        atom_mask=batch["atom_mask"],
        query_mask=batch["query_mask"],
    )
    assert out["sdf"].shape == (2, 8)
    assert out["features"].shape[:3] == (2, 8, expected_neighbors)
    assert out["mask"].shape == (2, 8, expected_neighbors)
    assert torch.all(out["sdf"][~batch["query_mask"]] == 0.0)
    assert torch.equal(out["mask"], out["mask"] & batch["query_mask"].unsqueeze(-1))


def test_global_feature_encoder_is_translation_invariant_with_atom_mask():
    encoder = GlobalFeatureEncoder(num_atom_types=16, atom_embed_dim=8, hidden_dim=32, out_dim=16)
    coords = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([[1, 6, 8, 0]], dtype=torch.long)
    radii = torch.tensor([[1.2, 1.3, 1.1, 0.0]], dtype=torch.float32)
    atom_mask = torch.tensor([[True, True, True, False]])
    shift = torch.tensor([[[3.2, -1.4, 0.7]]], dtype=torch.float32)

    base = encoder(coords, atom_types, radii, atom_mask=atom_mask)
    shifted = encoder(coords + shift, atom_types, radii, atom_mask=atom_mask)

    assert torch.allclose(base, shifted, atol=1e-5, rtol=1e-5)


def test_surface_model_from_config_builds_schnet_siren_variant():
    model = SurfaceModel.from_config(
        {
            "local_builder": {"atom_embed_dim": 8, "rbf_dim": 8, "cutoff": 6.0, "max_neighbors": 16},
            "local_encoder": {"type": "schnet", "hidden_dim": 32, "out_dim": 24, "num_layers": 2},
            "global_encoder": {"hidden_dim": 32, "out_dim": 20},
            "decoder": {"type": "siren", "hidden_dim": 48, "num_layers": 3},
            "position_encoding": {"enabled": True, "n_freq": 4},
        },
        num_atom_types=16,
    )
    dataset = MoleculeDataset(num_samples=1, num_query_points=8)
    sample = dataset[0]
    out = model(sample["coords"], sample["atom_types"], sample["radii"], sample["query_points"])
    assert out["sdf"].shape == (8,)
    assert out["z_local"].shape[-1] == 24
    assert out["z_global"].shape[-1] == 20


def test_surface_model_uses_charge_epsilon_sigma_features():
    torch.manual_seed(0)
    dataset = MoleculeDataset(num_samples=1, num_query_points=8)
    sample = dataset[0]
    model = SurfaceModel(num_atom_types=dataset.num_atom_types)

    base = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
        charges=sample["charges"],
        epsilon=sample["epsilon"],
        sigma=sample["sigma"],
    )["sdf"]
    perturbed = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
        charges=sample["charges"] + 1.0,
        epsilon=sample["epsilon"] + 0.5,
        sigma=sample["sigma"] + 0.25,
    )["sdf"]

    assert not torch.allclose(base, perturbed)


def test_surface_model_box_base_zero_init_matches_box_sdf():
    torch.manual_seed(0)
    dataset = MoleculeDataset(
        num_samples=1,
        num_query_points=8,
        initialization_mode="loose_box",
        loose_surface_padding=4.0,
        domain_padding=8.0,
    )
    sample = dataset[0]
    model = SurfaceModel(
        num_atom_types=dataset.num_atom_types,
        sdf_base_type="box",
        zero_init_output=True,
    )

    out = model(
        sample["coords"],
        sample["atom_types"],
        sample["radii"],
        sample["query_points"],
        bbox_lower=sample["surface_bbox_lower"],
        bbox_upper=sample["surface_bbox_upper"],
    )
    expected = box_sdf(sample["query_points"], sample["surface_bbox_lower"], sample["surface_bbox_upper"])

    assert torch.allclose(out["sdf"], expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["raw_residual"], torch.zeros_like(out["raw_residual"]), atol=1e-6, rtol=1e-6)


def test_fourier_encoder_output_dimension():
    encoder = FourierEncoder(d_in=3, n_freq=6)
    x = torch.zeros((2, 5, 3), dtype=torch.float32)
    y = encoder(x)
    assert y.shape == (2, 5, 39)
