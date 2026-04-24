"""Inference script: load a trained checkpoint and extract molecular surfaces.

Usage (from project root)
--------------------------
python scripts/infer_mesh.py \
    --ckpt outputs/checkpoints/latest.pt \
    --config configs/experiment/my_train.yaml \
    --processed_sample_dir /path/to/processed/sample \
    --spacing_angstrom 0.5 \
    --block_voxel_size 64 \
    --output_dir outputs/meshes
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomol_surface_unsup.datasets.molecule_dataset import ATOM_TYPE_TO_ID, MoleculeDataset
from biomol_surface_unsup.datasets.sampling import _compute_bbox
from biomol_surface_unsup.inference import load_processed_molecule
from biomol_surface_unsup.inference.native_ops import make_grid_block, narrow_band_bbox
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.utils.config import load_infer_config
from biomol_surface_unsup.visualization.export_mesh import export_mesh
from biomol_surface_unsup.visualization.plot_slices import plot_slices


def _compute_grid_metadata(
    coords: torch.Tensor,
    radii: torch.Tensor,
    spacing_angstrom: float,
    padding: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    spacing_angstrom = float(spacing_angstrom)
    if spacing_angstrom <= 0.0:
        raise ValueError(f"spacing_angstrom must be positive, got {spacing_angstrom}")

    margin = float(radii.max().item()) + float(padding)
    lo = coords.min(dim=0).values.detach().cpu().numpy() - margin
    hi = coords.max(dim=0).values.detach().cpu().numpy() + margin

    grid_shape = tuple(
        max(2, int(np.ceil((hi[i] - lo[i]) / spacing_angstrom)) + 1)
        for i in range(3)
    )
    spacing = np.asarray(
        [
            ((hi[i] - lo[i]) / (grid_shape[i] - 1)) if grid_shape[i] > 1 else spacing_angstrom
            for i in range(3)
        ],
        dtype=np.float32,
    )
    return lo.astype(np.float32), spacing, grid_shape


def _iter_grid_blocks(
    grid_shape: tuple[int, int, int],
    block_voxel_size: int,
):
    block_size = max(1, int(block_voxel_size))
    for x0 in range(0, grid_shape[0], block_size):
        x1 = min(x0 + block_size, grid_shape[0])
        for y0 in range(0, grid_shape[1], block_size):
            y1 = min(y0 + block_size, grid_shape[1])
            for z0 in range(0, grid_shape[2], block_size):
                z1 = min(z0 + block_size, grid_shape[2])
                block_shape = (x1 - x0, y1 - y0, z1 - z0)
                yield (slice(x0, x1), slice(y0, y1), slice(z0, z1)), (x0, y0, z0), block_shape


def _num_blocks(grid_shape: tuple[int, int, int], block_voxel_size: int) -> int:
    block_size = max(1, int(block_voxel_size))
    return (
        math.ceil(grid_shape[0] / block_size)
        * math.ceil(grid_shape[1] / block_size)
        * math.ceil(grid_shape[2] / block_size)
    )


def _predict_sdf_block(
    model: SurfaceModel,
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    radii: torch.Tensor,
    query_points: torch.Tensor,
    device: torch.device,
    batch_size: int,
    bbox_lower: torch.Tensor | None = None,
    bbox_upper: torch.Tensor | None = None,
) -> torch.Tensor:
    total = int(query_points.shape[0])
    predictions = torch.empty((total,), dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            model_kwargs = {}
            if str(getattr(model, "sdf_base_type", "none")).lower() == "box":
                model_kwargs = {"bbox_lower": bbox_lower, "bbox_upper": bbox_upper}
            out = model(coords, atom_types, radii, query_points[start:end], **model_kwargs)
            predictions[start:end] = out["sdf"].reshape(-1)
    return predictions


def _resolve_single_sample_dir(infer_cfg: dict[str, object]) -> Path | None:
    processed_sample_dir = infer_cfg.get("processed_sample_dir")
    if processed_sample_dir:
        return Path(str(processed_sample_dir))

    pdb_file = infer_cfg.get("pdb_file")
    chain_id = infer_cfg.get("chain_id")
    if not pdb_file:
        return None
    if not chain_id:
        raise ValueError("provide --chain_id together with --pdb_file for single-protein inference")

    from preprocess import process_one_pdb

    preprocess_dir = Path(str(infer_cfg.get("preprocess_dir", "outputs/infer_processed")))
    process_one_pdb(str(pdb_file), str(chain_id), str(preprocess_dir))
    return preprocess_dir / Path(str(pdb_file)).stem


def _expand_bbox(
    bbox: tuple[int, int, int, int, int, int] | None,
    grid_shape: tuple[int, int, int],
    halo: int = 1,
) -> tuple[int, int, int, int, int, int] | None:
    if bbox is None:
        return None
    x0, x1, y0, y1, z0, z1 = bbox
    return (
        max(0, x0 - halo),
        min(grid_shape[0], x1 + halo),
        max(0, y0 - halo),
        min(grid_shape[1], y1 + halo),
        max(0, z0 - halo),
        min(grid_shape[2], z1 + halo),
    )


def _merge_bbox(
    current: tuple[int, int, int, int, int, int] | None,
    local: tuple[int, int, int, int, int, int] | None,
    start_indices: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int] | None:
    if local is None:
        return current

    global_local = (
        local[0] + start_indices[0],
        local[1] + start_indices[0],
        local[2] + start_indices[1],
        local[3] + start_indices[1],
        local[4] + start_indices[2],
        local[5] + start_indices[2],
    )
    if current is None:
        return global_local
    return (
        min(current[0], global_local[0]),
        max(current[1], global_local[1]),
        min(current[2], global_local[2]),
        max(current[3], global_local[3]),
        min(current[4], global_local[4]),
        max(current[5], global_local[5]),
    )


def _write_grid_metadata(
    path: Path,
    lo: np.ndarray,
    spacing: np.ndarray,
    grid_shape: tuple[int, int, int],
    spacing_angstrom: float,
) -> None:
    payload = {
        "origin_angstrom": [float(v) for v in lo.tolist()],
        "spacing_angstrom": [float(v) for v in spacing.tolist()],
        "grid_shape": [int(v) for v in grid_shape],
        "target_spacing_angstrom": float(spacing_angstrom),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_blockwise_grid_inference(
    model: SurfaceModel,
    coords: torch.Tensor,
    atom_types: torch.Tensor,
    radii: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    lo: np.ndarray,
    spacing: np.ndarray,
    grid_shape: tuple[int, int, int],
    block_voxel_size: int,
    output_path: Path,
    narrow_band_width: float,
    track_narrow_band: bool,
    use_native_ops: bool,
    surface_bbox_lower: torch.Tensor | None = None,
    surface_bbox_upper: torch.Tensor | None = None,
) -> tuple[float, float, tuple[int, int, int, int, int, int] | None]:
    total_blocks = _num_blocks(grid_shape, block_voxel_size)
    lo_tensor = torch.as_tensor(lo, dtype=torch.float32, device=device)
    spacing_tensor = torch.as_tensor(spacing, dtype=torch.float32, device=device)
    sdf_memmap = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=grid_shape)

    sdf_min = float("inf")
    sdf_max = float("-inf")
    band_bbox = None

    for block_idx, (grid_slices, start_indices, block_shape) in enumerate(
        _iter_grid_blocks(grid_shape, block_voxel_size),
        start=1,
    ):
        query_block = make_grid_block(
            lo_tensor,
            spacing_tensor,
            start_indices,
            block_shape,
            use_native_ops=use_native_ops,
        )
        sdf_block = _predict_sdf_block(
            model,
            coords,
            atom_types,
            radii,
            query_block,
            device,
            batch_size,
            bbox_lower=surface_bbox_lower,
            bbox_upper=surface_bbox_upper,
        ).reshape(block_shape)

        sdf_min = min(sdf_min, float(sdf_block.min().item()))
        sdf_max = max(sdf_max, float(sdf_block.max().item()))

        if track_narrow_band:
            local_bbox = narrow_band_bbox(
                sdf_block,
                narrow_band_width,
                use_native_ops=use_native_ops,
            )
            band_bbox = _merge_bbox(band_bbox, local_bbox, start_indices)

        sdf_memmap[grid_slices] = sdf_block.detach().cpu().numpy()

        if block_idx == 1 or block_idx == total_blocks or (block_idx % 10 == 0):
            print(
                f"[infer_mesh]   block {block_idx}/{total_blocks} "
                f"written, sdf range so far: [{sdf_min:.3f}, {sdf_max:.3f}]"
            )

    sdf_memmap.flush()
    del sdf_memmap
    return sdf_min, sdf_max, band_bbox


def _marching_cubes(
    sdf_grid: np.ndarray,
    lo: np.ndarray,
    spacing: np.ndarray,
    level: float,
) -> dict[str, np.ndarray] | None:
    try:
        from skimage.measure import marching_cubes as skimage_mc
    except ImportError:
        print(
            "[infer_mesh] WARNING: scikit-image not installed — skipping mesh extraction.\n"
            "             Install with: pip install scikit-image"
        )
        return None
    sdf_grid = np.array(sdf_grid, dtype=np.float32, copy=True, order="C")
    sdf_min = float(np.min(sdf_grid))
    sdf_max = float(np.max(sdf_grid))
    if not (sdf_min <= level <= sdf_max):
        print(
            "[infer_mesh] no surface extracted: SDF does not cross the requested level "
            f"(level={level:.6f}, min={sdf_min:.6f}, max={sdf_max:.6f})"
        )
        return None

    try:
        verts_vox, faces, _normals, _vals = skimage_mc(sdf_grid, level=level)
    except ValueError as exc:
        print(
            "[infer_mesh] marching_cubes failed after level-crossing check: "
            f"{exc} (level={level:.6f}, min={sdf_min:.6f}, max={sdf_max:.6f})"
        )
        return None
    verts_world = verts_vox * spacing + lo
    return {"verts": verts_world.astype(np.float32), "faces": faces.astype(np.int32)}


def _filter_mesh_components(
    mesh: dict[str, np.ndarray],
    min_component_faces: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, int]]:
    faces = mesh["faces"]
    verts = mesh["verts"]
    num_faces = int(faces.shape[0])
    if num_faces == 0:
        return None, {
            "num_components": 0,
            "kept_faces": 0,
            "dropped_small_components": 0,
            "kept_component_faces": 0,
        }

    vertex_to_faces: dict[int, list[int]] = {}
    for face_idx, face in enumerate(faces):
        for vertex_idx in face.tolist():
            vertex_to_faces.setdefault(int(vertex_idx), []).append(face_idx)

    visited = np.zeros(num_faces, dtype=bool)
    components: list[list[int]] = []
    for seed_face in range(num_faces):
        if visited[seed_face]:
            continue
        stack = [seed_face]
        visited[seed_face] = True
        component: list[int] = []
        while stack:
            face_idx = stack.pop()
            component.append(face_idx)
            for vertex_idx in faces[face_idx].tolist():
                for neighbor_face in vertex_to_faces[int(vertex_idx)]:
                    if not visited[neighbor_face]:
                        visited[neighbor_face] = True
                        stack.append(neighbor_face)
        components.append(component)

    eligible_components = [
        component for component in components if len(component) >= int(min_component_faces)
    ]
    if not eligible_components:
        return None, {
            "num_components": len(components),
            "kept_faces": 0,
            "dropped_small_components": len(components),
            "kept_component_faces": 0,
        }

    largest_component = max(eligible_components, key=len)
    kept_face_indices = np.asarray(sorted(largest_component), dtype=np.int32)
    kept_faces = faces[kept_face_indices]
    kept_vertex_indices = np.unique(kept_faces.reshape(-1))
    remap = np.full(int(verts.shape[0]), -1, dtype=np.int32)
    remap[kept_vertex_indices] = np.arange(kept_vertex_indices.shape[0], dtype=np.int32)
    filtered_mesh = {
        "verts": verts[kept_vertex_indices].astype(np.float32, copy=False),
        "faces": remap[kept_faces].astype(np.int32, copy=False),
    }
    return filtered_mesh, {
        "num_components": len(components),
        "kept_faces": int(filtered_mesh["faces"].shape[0]),
        "dropped_small_components": len(components) - len(eligible_components),
        "kept_component_faces": len(largest_component),
    }


def main() -> None:
    cfg = load_infer_config()
    infer_cfg = cfg["infer"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    if infer_cfg["device"] is not None:
        device = torch.device(infer_cfg["device"])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[infer_mesh] using device: {device}")

    split = str(infer_cfg["split"])
    single_sample_dir = _resolve_single_sample_dir(infer_cfg)
    dataset = None
    samples: list[dict[str, object]] = []
    if single_sample_dir is not None:
        molecule = load_processed_molecule(single_sample_dir)
        samples = [
            {
                "id": single_sample_dir.name,
                "coords": molecule["coords"],
                "atom_types": molecule["atom_types"],
                "radii": molecule["radii"],
            }
        ]
        print(f"[infer_mesh] single sample mode: {single_sample_dir}")
    else:
        num_samples = infer_cfg["num_samples"]
        dataset = MoleculeDataset(
            root=data_cfg.get("root", "data/processed"),
            split=split,
            num_samples=num_samples,
            num_atoms=int(data_cfg.get("num_atoms", 4)),
            num_query_points=int(data_cfg.get("num_query_points", 512)),
            bbox_padding=float(data_cfg.get("bbox_padding", 4.0)),
            initialization_mode=str(data_cfg.get("initialization_mode", "tight_atomic")),
            loose_surface_padding=float(
                data_cfg.get("loose_surface_padding", data_cfg.get("bbox_padding", 4.0))
            ),
            domain_padding=float(data_cfg.get("domain_padding", data_cfg.get("bbox_padding", 4.0))),
            containment_jitter=float(data_cfg.get("containment_jitter", 0.15)),
            surface_band_width=float(data_cfg.get("surface_band_width", data_cfg.get("surface_bandwidth", 0.5))),
        )
        samples = [dataset[idx] for idx in range(len(dataset))]
        print(f"[infer_mesh] split='{split}', found {len(samples)} samples")

    num_atom_types = len(ATOM_TYPE_TO_ID) if dataset is None else dataset.num_atom_types
    model = SurfaceModel.from_config(model_cfg, num_atom_types=num_atom_types)
    model.to(device)

    ckpt_path = Path(infer_cfg["ckpt"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw_ckpt, dict) and "model" in raw_ckpt:
        state_dict = raw_ckpt["model"]
        saved_epoch = raw_ckpt.get("epoch", "?")
        print(f"[infer_mesh] loaded training checkpoint from epoch {saved_epoch}: {ckpt_path}")
    elif isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt:
        state_dict = raw_ckpt["model_state_dict"]
        saved_epoch = raw_ckpt.get("epoch", "?")
        print(f"[infer_mesh] loaded checkpoint from epoch {saved_epoch}: {ckpt_path}")
    else:
        state_dict = raw_ckpt
        print(f"[infer_mesh] loaded raw state dict from: {ckpt_path}")
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = Path(infer_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    spacing_angstrom = float(infer_cfg.get("spacing_angstrom", 0.1))
    block_voxel_size = int(infer_cfg.get("block_voxel_size", 64))
    batch_size = int(infer_cfg["batch_size"])
    loose_surface_padding = float(data_cfg.get("loose_surface_padding", data_cfg.get("bbox_padding", 4.0)))
    padding = float(data_cfg.get("domain_padding", data_cfg.get("bbox_padding", 4.0)))
    narrow_band_width = float(infer_cfg.get("narrow_band_width", 2.0))
    isosurface_level = float(infer_cfg.get("isosurface_level", 0.0))
    min_component_faces = int(infer_cfg.get("min_component_faces", 0))
    narrow_band_crop = bool(infer_cfg.get("narrow_band_crop", True))
    use_native_ops = bool(infer_cfg.get("use_native_ops", True))

    for idx, sample in enumerate(samples):
        mol_id = str(sample.get("id", f"sample_{idx}"))
        print(f"\n[infer_mesh] [{idx+1}/{len(samples)}] molecule: {mol_id}")

        coords = sample["coords"].to(device)
        atom_types = sample["atom_types"].to(device)
        radii = sample["radii"].to(device)
        surface_bbox_lower = sample.get("surface_bbox_lower")
        surface_bbox_upper = sample.get("surface_bbox_upper")
        if surface_bbox_lower is None or surface_bbox_upper is None:
            surface_bbox_lower, surface_bbox_upper = _compute_bbox(coords, radii, loose_surface_padding)
        surface_bbox_lower = torch.as_tensor(surface_bbox_lower, dtype=torch.float32, device=device)
        surface_bbox_upper = torch.as_tensor(surface_bbox_upper, dtype=torch.float32, device=device)

        lo, spacing, grid_shape = _compute_grid_metadata(coords, radii, spacing_angstrom, padding)
        print(
            f"[infer_mesh]   grid shape: {grid_shape}, spacing(Å): {tuple(float(v) for v in spacing)}, "
            f"block_voxel_size={block_voxel_size}"
        )

        npy_path = out_dir / f"{mol_id}_sdf.npy"
        meta_path = out_dir / f"{mol_id}_sdf_meta.json"
        _write_grid_metadata(meta_path, lo, spacing, grid_shape, spacing_angstrom)

        sdf_min, sdf_max, band_bbox = _run_blockwise_grid_inference(
            model,
            coords,
            atom_types,
            radii,
            device=device,
            batch_size=batch_size,
            lo=lo,
            spacing=spacing,
            grid_shape=grid_shape,
            block_voxel_size=block_voxel_size,
            output_path=npy_path,
            narrow_band_width=narrow_band_width,
            track_narrow_band=bool(infer_cfg["extract_mesh"]) and narrow_band_crop,
            use_native_ops=use_native_ops,
            surface_bbox_lower=surface_bbox_lower,
            surface_bbox_upper=surface_bbox_upper,
        )
        print(f"[infer_mesh]   SDF range: [{sdf_min:.3f}, {sdf_max:.3f}]")
        print(f"[infer_mesh]   SDF grid saved → {npy_path}")
        print(f"[infer_mesh]   metadata saved → {meta_path}")

        sdf_grid = np.load(npy_path, mmap_mode="r")

        if infer_cfg["extract_mesh"]:
            crop_bbox = _expand_bbox(band_bbox, grid_shape, halo=1) if narrow_band_crop else None
            if crop_bbox is None and narrow_band_crop:
                print("[infer_mesh]   (no narrow band found near the zero level — skipping mesh extraction)")
            else:
                crop_lo = lo
                crop_grid = np.array(sdf_grid, dtype=np.float32, copy=True, order="C")
                if crop_bbox is not None:
                    x0, x1, y0, y1, z0, z1 = crop_bbox
                    crop_grid = np.array(
                        sdf_grid[x0:x1, y0:y1, z0:z1],
                        dtype=np.float32,
                        copy=True,
                        order="C",
                    )
                    crop_lo = lo + spacing * np.asarray([x0, y0, z0], dtype=np.float32)
                    print(
                        "[infer_mesh]   marching cubes crop bbox="
                        f"{crop_bbox}, crop shape={crop_grid.shape}"
                    )
                mesh = _marching_cubes(
                    crop_grid,
                    crop_lo,
                    spacing,
                    level=isosurface_level,
                )
                if mesh is not None:
                    filtered_mesh, filter_stats = _filter_mesh_components(
                        mesh,
                        min_component_faces=min_component_faces,
                    )
                    if filtered_mesh is None:
                        print(
                            "[infer_mesh]   mesh filtering removed all components "
                            f"(components={filter_stats['num_components']}, "
                            f"min_component_faces={min_component_faces})"
                        )
                        mesh = None
                    else:
                        mesh = filtered_mesh
                        print(
                            "[infer_mesh]   kept largest mesh shell "
                            f"(components={filter_stats['num_components']}, "
                            f"kept_faces={filter_stats['kept_faces']}, "
                            f"dropped_small_components={filter_stats['dropped_small_components']}, "
                            f"min_component_faces={min_component_faces})"
                        )
                if mesh is not None:
                    mesh_path = out_dir / f"{mol_id}_surface.obj"
                    export_mesh(mesh, mesh_path)
                    print(
                        f"[infer_mesh]   mesh saved → {mesh_path} "
                        f"({len(mesh['verts'])} verts, {len(mesh['faces'])} faces)"
                    )
                else:
                    print("[infer_mesh]   (no surface extracted — SDF may not cross zero)")

        if infer_cfg["plot_slices"]:
            slice_path = out_dir / f"{mol_id}_slices.png"
            try:
                plot_slices(
                    sdf_grid,
                    output_path=slice_path,
                    axis=2,
                    num_slices=4,
                    molecule_id=mol_id,
                )
                print(f"[infer_mesh]   slices saved → {slice_path}")
            except ImportError as exc:
                print(f"[infer_mesh]   WARNING: {exc}")

    print(f"\n[infer_mesh] Done. All outputs in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
