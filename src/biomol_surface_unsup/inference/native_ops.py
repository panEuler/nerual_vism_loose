from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


def _python_make_grid_block(
    lo: torch.Tensor,
    spacing: torch.Tensor,
    start_indices: tuple[int, int, int],
    block_shape: tuple[int, int, int],
) -> torch.Tensor:
    x = torch.arange(start_indices[0], start_indices[0] + block_shape[0], device=lo.device, dtype=lo.dtype)
    y = torch.arange(start_indices[1], start_indices[1] + block_shape[1], device=lo.device, dtype=lo.dtype)
    z = torch.arange(start_indices[2], start_indices[2] + block_shape[2], device=lo.device, dtype=lo.dtype)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([gx, gy, gz], dim=-1)
    return lo + grid.reshape(-1, 3) * spacing


def _python_narrow_band_bbox(
    sdf_block: torch.Tensor,
    threshold: float,
) -> tuple[int, int, int, int, int, int] | None:
    mask = torch.abs(sdf_block) <= float(threshold)
    if not torch.any(mask):
        return None
    coords = torch.nonzero(mask, as_tuple=False)
    mins = coords.amin(dim=0)
    maxs = coords.amax(dim=0) + 1
    return (
        int(mins[0].item()),
        int(maxs[0].item()),
        int(mins[1].item()),
        int(maxs[1].item()),
        int(mins[2].item()),
        int(maxs[2].item()),
    )


@lru_cache(maxsize=1)
def _load_native_extension():
    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except Exception:
        return None

    source_root = Path(__file__).resolve().parents[1] / "csrc"
    cpp_source = source_root / "infer_mesh_native.cpp"
    cuda_source = source_root / "infer_mesh_native_cuda.cu"
    sources = [str(cpp_source)]
    extra_cuda_sources: list[str] = []
    with_cuda = bool(torch.cuda.is_available() and CUDA_HOME and cuda_source.exists())
    if with_cuda:
        extra_cuda_sources.append(str(cuda_source))

    try:
        build_directory = Path(os.environ.get("TMPDIR", "/tmp")) / "biomol_surface_unsup_native_ops"
        build_directory.mkdir(parents=True, exist_ok=True)
        return load(
            name="biomol_surface_unsup_infer_mesh_native",
            sources=sources + extra_cuda_sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"] if with_cuda else [],
            build_directory=str(build_directory),
            verbose=False,
        )
    except Exception:
        return None


def make_grid_block(
    lo: torch.Tensor,
    spacing: torch.Tensor,
    start_indices: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    *,
    use_native_ops: bool = True,
) -> torch.Tensor:
    ext = _load_native_extension() if use_native_ops else None
    if ext is not None:
        return ext.make_grid_block(
            lo,
            spacing,
            [int(v) for v in start_indices],
            [int(v) for v in block_shape],
        )
    return _python_make_grid_block(lo, spacing, start_indices, block_shape)


def narrow_band_bbox(
    sdf_block: torch.Tensor,
    threshold: float,
    *,
    use_native_ops: bool = True,
) -> tuple[int, int, int, int, int, int] | None:
    ext = _load_native_extension() if use_native_ops else None
    if ext is not None:
        bbox = ext.narrow_band_bbox(sdf_block, float(threshold))
        values = [int(v) for v in bbox.tolist()]
        if values[0] < 0:
            return None
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    return _python_narrow_band_bbox(sdf_block, threshold)
