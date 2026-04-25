from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - fallback when torch is unavailable
    class Dataset:  # type: ignore[override]
        pass

try:
    import torch
except Exception:  # pragma: no cover - fallback when torch is unavailable
    torch = None

from biomol_surface_unsup.datasets.sampling import sample_query_points


ATOM_TYPE_TO_ID = {
    "PAD": 0,
    "UNK": 1,
    "H": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "S": 6,
    "P": 7,
    "F": 8,
    "CL": 9,
    "BR": 10,
    "I": 11,
    "MG": 12,
    "CA": 13,
    "ZN": 14,
    "FE": 15,
}

_AUTO_SPLIT_RATIOS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
_AUTO_SPLIT_ORDER = ("train", "val", "test")
_DEFAULT_SPLIT_SEED = 42


@dataclass(frozen=True)
class MoleculeRecord:
    sample_id: str
    directory: Path
    prefix: str


def _normalize_atom_type(atom_type: str) -> str:
    token = str(atom_type).strip().upper()
    return token if token else "UNK"


def _encode_atom_types(atom_types: np.ndarray) -> np.ndarray:
    encoded = [ATOM_TYPE_TO_ID.get(_normalize_atom_type(atom_type), ATOM_TYPE_TO_ID["UNK"]) for atom_type in atom_types]
    return np.asarray(encoded, dtype=np.int64)


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def _find_sample_prefix(sample_dir: Path) -> str:
    coords_files = sorted(sample_dir.glob("*_coords.npy"))
    if len(coords_files) != 1:
        raise FileNotFoundError(f"expected exactly one '*_coords.npy' in {sample_dir}, found {len(coords_files)}")
    return coords_files[0].name[: -len("_coords.npy")]


def _required_sample_fields(prefix: str) -> dict[str, str]:
    return {
        "coords": f"{prefix}_coords.npy",
        "atom_types": f"{prefix}_atom_types.npy",
        "radii": f"{prefix}_radii.npy",
        "charges": f"{prefix}_charges.npy",
        "epsilon": f"{prefix}_epsilon.npy",
        "sigma": f"{prefix}_sigma.npy",
        "res_ids": f"{prefix}_res_ids.npy",
    }


class MoleculeDataset(Dataset):
    """Dataset for processed protein-chain directories under `data/processed`.

    Per-sample tensor shapes before collation:
    - coords: [N, 3]
    - atom_types: [N]
    - radii: [N]
    - charges: [N]
    - epsilon: [N]
    - sigma: [N]
    - res_ids: [N]
    - query_points: [Q, 3]
    - query_group: [Q]
    - containment_points: [C, 3]
    """

    def __init__(
        self,
        root: str = "data/processed",
        split: str = "train",
        num_samples: int | None = None,
        num_atoms: int | None = None,
        num_query_points: int = 512,
        bbox_padding: float = 4.0,
        initialization_mode: str = "tight_atomic",
        loose_surface_padding: float | None = None,
        domain_padding: float | None = None,
        containment_jitter: float = 0.15,
        surface_band_width: float = 0.25,
        num_area_points: int = 0,
        split_seed: int = _DEFAULT_SPLIT_SEED,
    ) -> None:
        del num_atoms  # Real processed samples determine atom count from disk.
        self.root = Path(root)
        self.split = split
        self.split_seed = int(split_seed)
        self.num_query_points = int(num_query_points)
        self.bbox_padding = float(bbox_padding)
        self.initialization_mode = str(initialization_mode)
        self.loose_surface_padding = (
            float(self.bbox_padding) if loose_surface_padding is None else float(loose_surface_padding)
        )
        self.domain_padding = float(self.bbox_padding) if domain_padding is None else float(domain_padding)
        self.containment_jitter = float(containment_jitter)
        self.surface_band_width = float(surface_band_width)
        self.num_area_points = int(num_area_points)
        self.num_atom_types = len(ATOM_TYPE_TO_ID)
        self.records = self._discover_records()
        if num_samples is not None:
            self.records = self.records[: int(num_samples)]
        if not self.records:
            raise FileNotFoundError(f"no processed molecule samples found under {self.root}")

    def _discover_records(self) -> list[MoleculeRecord]:
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        split_file = self.root / f"{self.split}.txt"
        if split_file.exists():
            sample_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if self.root.name == "processed":
                split_root = self.root.parent / self.split
                sample_dirs = [split_root / sample_id for sample_id in sample_ids]
            else:
                sample_dirs = [self.root / sample_id for sample_id in sample_ids]
        elif any(self.root.glob("*_coords.npy")):
            sample_dirs = [self.root]
        else:
            sample_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
            sample_dirs = self._select_split_dirs(sample_dirs)

        records: list[MoleculeRecord] = []
        for sample_dir in sample_dirs:
            prefix = _find_sample_prefix(sample_dir)
            required = _required_sample_fields(prefix)
            missing = [name for name, filename in required.items() if not (sample_dir / filename).exists()]
            if missing:
                raise FileNotFoundError(f"sample {sample_dir.name} is missing required fields: {', '.join(missing)}")
            records.append(MoleculeRecord(sample_id=sample_dir.name, directory=sample_dir, prefix=prefix))
        return records

    def _select_split_dirs(self, sample_dirs: list[Path]) -> list[Path]:
        if self.split not in _AUTO_SPLIT_RATIOS or len(sample_dirs) <= 1:
            return sample_dirs
        if self.root.name == self.split:
            return sample_dirs

        split_dirs = self._partition_sample_dirs(sample_dirs)
        if self.root.name == "processed":
            return self._materialize_split_dirs(split_dirs)
        return split_dirs[self.split]

    def _partition_sample_dirs(self, sample_dirs: list[Path]) -> dict[str, list[Path]]:
        rng = np.random.default_rng(self.split_seed)
        shuffled_dirs = [sample_dirs[idx] for idx in rng.permutation(len(sample_dirs))]
        split_counts = self._compute_split_counts(len(shuffled_dirs))

        train_end = split_counts["train"]
        val_end = train_end + split_counts["val"]
        split_dict = {
            "train": shuffled_dirs[:train_end],
            "val": shuffled_dirs[train_end:val_end],
            "test": shuffled_dirs[val_end:],
        }
        
        # Save split info to files so it doesn't partition again on subsequent runs
        try:
            for split_name, dirs in split_dict.items():
                split_file = self.root / f"{split_name}.txt"
                if not split_file.exists():
                    split_file.write_text("\n".join(d.name for d in dirs) + "\n", encoding="utf-8")
        except OSError:
            pass  # Fail gracefully if root is read-only

        return split_dict

    def _materialize_split_dirs(self, split_dirs: dict[str, list[Path]]) -> list[Path]:
        split_root = self.root.parent / self.split
        split_root.mkdir(parents=True, exist_ok=True)

        for split_name, source_dirs in split_dirs.items():
            current_split_root = self.root.parent / split_name
            current_split_root.mkdir(parents=True, exist_ok=True)
            for source_dir in source_dirs:
                link_dir = current_split_root / source_dir.name
                if link_dir.is_symlink():
                    if link_dir.resolve() == source_dir.resolve():
                        continue
                    link_dir.unlink()
                elif link_dir.exists():
                    continue
                os.symlink(source_dir.resolve(), link_dir, target_is_directory=True)

        return [split_root / source_dir.name for source_dir in split_dirs[self.split]]

    def _compute_split_counts(self, num_samples: int) -> dict[str, int]:
        desired_counts = {
            split_name: num_samples * split_ratio for split_name, split_ratio in _AUTO_SPLIT_RATIOS.items()
        }
        split_counts = {
            split_name: math.floor(desired_count) for split_name, desired_count in desired_counts.items()
        }

        remaining = num_samples - sum(split_counts.values())
        if remaining > 0:
            split_priority = sorted(
                _AUTO_SPLIT_ORDER,
                key=lambda split_name: (desired_counts[split_name] - split_counts[split_name]),
                reverse=True,
            )
            for split_name in split_priority[:remaining]:
                split_counts[split_name] += 1
        return split_counts

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("MoleculeDataset requires torch in the current training path")

        record = self.records[idx]
        required = _required_sample_fields(record.prefix)
        arrays = {name: _load_npy(record.directory / filename) for name, filename in required.items()}

        coords = torch.as_tensor(arrays["coords"], dtype=torch.float32)
        radii = torch.as_tensor(arrays["radii"], dtype=torch.float32)
        charges = torch.as_tensor(arrays["charges"], dtype=torch.float32)
        epsilon = torch.as_tensor(arrays["epsilon"], dtype=torch.float32)
        sigma = torch.as_tensor(arrays["sigma"], dtype=torch.float32)
        res_ids = torch.as_tensor(arrays["res_ids"], dtype=torch.long)
        atom_types = torch.as_tensor(_encode_atom_types(arrays["atom_types"]), dtype=torch.long)

        sampling = sample_query_points(
            coords=coords,
            num_query_points=self.num_query_points,
            padding=self.bbox_padding,
            radii=radii,
            containment_jitter=self.containment_jitter,
            surface_band_width=self.surface_band_width,
            num_area_points=self.num_area_points,
            initialization_mode=self.initialization_mode,
            loose_surface_padding=self.loose_surface_padding,
            domain_padding=self.domain_padding,
        )

        return {
            "id": record.sample_id,
            "coords": coords,
            "atom_types": atom_types,
            "radii": radii,
            "charges": charges,
            "epsilon": epsilon,
            "sigma": sigma,
            "res_ids": res_ids,
            "query_points": sampling["query_points"],
            "query_group": sampling["query_group"],
            "containment_points": sampling["containment_points"],
            "surface_bbox_lower": sampling["surface_bbox_lower"],
            "surface_bbox_upper": sampling["surface_bbox_upper"],
            "domain_bbox_lower": sampling["domain_bbox_lower"],
            "domain_bbox_upper": sampling["domain_bbox_upper"],
            "bbox_lower": sampling["bbox_lower"],
            "bbox_upper": sampling["bbox_upper"],
            "bbox_volume": sampling["bbox_volume"],
            "sampling_counts": sampling["sampling_counts"],
        }
