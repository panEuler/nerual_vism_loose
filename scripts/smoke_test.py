from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
except Exception as exc:  # pragma: no cover - script path
    raise SystemExit(f"smoke_test requires torch: {exc}")

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.losses.loss_builder import build_loss_fn
from biomol_surface_unsup.models.surface_model import SurfaceModel


def main() -> int:
    dataset = MoleculeDataset(num_samples=2, num_atoms=4, num_query_points=16)
    batch = collate_fn([dataset[0], dataset[1]])

    assert batch["coords"].shape[0] == 2
    assert batch["coords"].shape[-1] == 3
    assert batch["atom_mask"].shape[:2] == batch["coords"].shape[:2]
    assert batch["query_points"].shape[0] == 2
    assert batch["query_points"].shape[-1] == 3
    assert batch["query_mask"].shape[:2] == batch["query_points"].shape[:2]

    model = SurfaceModel(num_atom_types=16)
    query_points = batch["query_points"].requires_grad_(True)
    output = model(
        batch["coords"],
        batch["atom_types"],
        batch["radii"],
        query_points,
        atom_mask=batch["atom_mask"],
        query_mask=batch["query_mask"],
    )
    assert "sdf" in output
    assert output["sdf"].shape == batch["query_group"].shape
    assert output["features"].shape[:2] == batch["query_points"].shape[:2]
    assert output["mask"].shape[:2] == batch["query_points"].shape[:2]

    loss_fn = build_loss_fn({"loss": {}})
    losses = loss_fn(
        {
            "coords": batch["coords"],
            "atom_types": batch["atom_types"],
            "radii": batch["radii"],
            "atom_mask": batch["atom_mask"],
            "query_points": query_points,
            "query_group": batch["query_group"],
            "query_mask": batch["query_mask"],
            "containment_points": batch["containment_points"],
            "containment_mask": batch["containment_mask"],
            "bbox_lower": batch["bbox_lower"],
            "bbox_upper": batch["bbox_upper"],
            "bbox_volume": batch["bbox_volume"],
        },
        output,
    )
    assert "total" in losses
    assert torch.isfinite(losses["total"])
    print("smoke test ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
