from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import ATOM_TYPE_TO_ID, MoleculeDataset


def _write_processed_sample(sample_dir: Path, prefix: str, num_atoms: int) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / f"{prefix}_coords.npy", np.stack([np.arange(num_atoms), np.zeros(num_atoms), np.zeros(num_atoms)], axis=-1))
    atom_types = np.array(["C", "N", "O", "S"][:num_atoms], dtype="<U2")
    np.save(sample_dir / f"{prefix}_atom_types.npy", atom_types)
    np.save(sample_dir / f"{prefix}_radii.npy", np.linspace(1.2, 1.5, num_atoms))
    np.save(sample_dir / f"{prefix}_charges.npy", np.linspace(-0.2, 0.2, num_atoms))
    np.save(sample_dir / f"{prefix}_epsilon.npy", np.linspace(0.1, 0.4, num_atoms))
    np.save(sample_dir / f"{prefix}_sigma.npy", np.linspace(2.0, 2.3, num_atoms))
    np.save(sample_dir / f"{prefix}_res_ids.npy", np.arange(1, num_atoms + 1, dtype=np.int32))
    np.save(sample_dir / f"{prefix}_atom_names.npy", np.array(["CA", "N", "O", "CB"][:num_atoms], dtype="<U4"))
    np.save(sample_dir / f"{prefix}_res_names.npy", np.array(["GLY", "ALA", "SER", "VAL"][:num_atoms], dtype="<U3"))


def test_dataset_reads_processed_sample_with_sampling_metadata(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _write_processed_sample(tmp_path / "1ABC_TEST", "1ABC_TEST_A", num_atoms=4)
    dataset = MoleculeDataset(root=str(tmp_path), num_query_points=8)
    sample = dataset[0]

    assert sample["id"] == "1ABC_TEST"
    assert tuple(sample["coords"].shape) == (4, 3)
    assert tuple(sample["atom_types"].shape) == (4,)
    assert tuple(sample["radii"].shape) == (4,)
    assert tuple(sample["charges"].shape) == (4,)
    assert tuple(sample["epsilon"].shape) == (4,)
    assert tuple(sample["sigma"].shape) == (4,)
    assert tuple(sample["res_ids"].shape) == (4,)
    assert tuple(sample["query_points"].shape) == (8, 3)
    assert tuple(sample["query_group"].shape) == (8,)
    assert tuple(sample["containment_points"].shape) == (2, 3)
    assert tuple(sample["surface_bbox_lower"].shape) == (3,)
    assert tuple(sample["surface_bbox_upper"].shape) == (3,)
    assert tuple(sample["domain_bbox_lower"].shape) == (3,)
    assert tuple(sample["domain_bbox_upper"].shape) == (3,)
    assert sample["atom_types"].tolist() == [
        ATOM_TYPE_TO_ID["C"],
        ATOM_TYPE_TO_ID["N"],
        ATOM_TYPE_TO_ID["O"],
        ATOM_TYPE_TO_ID["S"],
    ]
    assert sample["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2}


def test_collate_fn_pads_atoms_and_queries_without_dropping_samples(tmp_path: Path) -> None:
    torch.manual_seed(0)
    _write_processed_sample(tmp_path / "1ABC_TEST", "1ABC_TEST_A", num_atoms=4)
    _write_processed_sample(tmp_path / "2XYZ_TEST", "2XYZ_TEST_A", num_atoms=3)
    dataset = MoleculeDataset(root=str(tmp_path), num_query_points=8)
    batch = collate_fn([dataset[0], dataset[1]])

    assert set(batch["id"]) == {"1ABC_TEST", "2XYZ_TEST"}
    assert tuple(batch["coords"].shape) == (2, 4, 3)
    assert tuple(batch["atom_types"].shape) == (2, 4)
    assert tuple(batch["radii"].shape) == (2, 4)
    assert tuple(batch["charges"].shape) == (2, 4)
    assert tuple(batch["epsilon"].shape) == (2, 4)
    assert tuple(batch["sigma"].shape) == (2, 4)
    assert tuple(batch["res_ids"].shape) == (2, 4)
    assert tuple(batch["atom_mask"].shape) == (2, 4)
    assert tuple(batch["query_points"].shape) == (2, 8, 3)
    assert tuple(batch["query_group"].shape) == (2, 8)
    assert tuple(batch["query_mask"].shape) == (2, 8)
    assert tuple(batch["surface_bbox_lower"].shape) == (2, 3)
    assert tuple(batch["surface_bbox_upper"].shape) == (2, 3)
    assert tuple(batch["domain_bbox_lower"].shape) == (2, 3)
    assert tuple(batch["domain_bbox_upper"].shape) == (2, 3)
    batch_index_by_id = {sample_id: idx for idx, sample_id in enumerate(batch["id"])}
    assert batch["atom_mask"][batch_index_by_id["1ABC_TEST"]].sum().item() == 4
    assert batch["atom_mask"][batch_index_by_id["2XYZ_TEST"]].sum().item() == 3
    assert batch["query_mask"][batch_index_by_id["1ABC_TEST"]].sum().item() == 8
    assert batch["query_mask"][batch_index_by_id["2XYZ_TEST"]].sum().item() == 8
    assert batch["sampling_counts"] == {"global": 8, "containment": 4, "surface_band": 4}


def test_dataset_auto_split_is_deterministic_with_train_val_test_ratio(tmp_path: Path) -> None:
    processed_root = tmp_path / "data" / "processed"
    for idx in range(10):
        sample_id = f"{idx:02d}ABC_CHAIN"
        _write_processed_sample(processed_root / sample_id, f"{sample_id}_A", num_atoms=4)

    train_dataset = MoleculeDataset(root=str(processed_root), split="train", num_query_points=8)
    val_dataset = MoleculeDataset(root=str(processed_root), split="val", num_query_points=8)
    test_dataset = MoleculeDataset(root=str(processed_root), split="test", num_query_points=8)
    train_dataset_again = MoleculeDataset(root=str(processed_root), split="train", num_query_points=8)
    val_dataset_again = MoleculeDataset(root=str(processed_root), split="val", num_query_points=8)
    test_dataset_again = MoleculeDataset(root=str(processed_root), split="test", num_query_points=8)

    train_ids = [record.sample_id for record in train_dataset.records]
    val_ids = [record.sample_id for record in val_dataset.records]
    test_ids = [record.sample_id for record in test_dataset.records]

    assert len(train_ids) == 8
    assert len(val_ids) == 1
    assert len(test_ids) == 1
    assert (processed_root.parent / "train").is_dir()
    assert (processed_root.parent / "val").is_dir()
    assert (processed_root.parent / "test").is_dir()
    assert all(record.directory.parent == processed_root.parent / "train" for record in train_dataset.records)
    assert all(record.directory.parent == processed_root.parent / "val" for record in val_dataset.records)
    assert all(record.directory.parent == processed_root.parent / "test" for record in test_dataset.records)
    assert all(record.directory.is_symlink() for record in train_dataset.records)
    assert all(record.directory.is_symlink() for record in val_dataset.records)
    assert all(record.directory.is_symlink() for record in test_dataset.records)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)
    assert set(train_ids) | set(val_ids) | set(test_ids) == {f"{idx:02d}ABC_CHAIN" for idx in range(10)}
    assert train_ids == [record.sample_id for record in train_dataset_again.records]
    assert val_ids == [record.sample_id for record in val_dataset_again.records]
    assert test_ids == [record.sample_id for record in test_dataset_again.records]
