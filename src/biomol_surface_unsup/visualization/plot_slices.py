"""SDF slice visualization: plot cross-sections of the predicted scalar field."""
from __future__ import annotations
from pathlib import Path
import numpy as np
def plot_slices(
    sdf_grid: np.ndarray,
    output_path: str | Path | None = None,
    *,
    axis: int = 2,
    num_slices: int = 4,
    molecule_id: str = "",
) -> dict[str, int]:
    """Render cross-sectional slices through a 3D SDF volume.
    Args:
        sdf_grid: Float array of shape (R, R, R) with predicted SDF values.
        output_path: If given, save the figure to this path (PNG/PDF).
        axis: Axis along which to slice (0=X, 1=Y, 2=Z).
        num_slices: Number of evenly-spaced slices to show.
        molecule_id: Optional title annotation.
    Returns:
        A dict with metadata: ``{"num_slices": n, "shape": list}``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_slices — install with: pip install matplotlib"
        ) from exc
    grid = np.asarray(sdf_grid)
    assert grid.ndim == 3, f"Expected 3D array, got shape {grid.shape}"
    size = grid.shape[axis]
    indices = np.linspace(0, size - 1, num_slices + 2, dtype=int)[1:-1]  # skip edges
    fig, axes = plt.subplots(1, num_slices, figsize=(4 * num_slices, 4))
    if num_slices == 1:
        axes = [axes]
    axis_labels = ["X", "Y", "Z"]
    vmin, vmax = float(grid.min()), float(grid.max())
    abs_max = max(abs(vmin), abs(vmax), 1e-6)
    for ax, idx in zip(axes, indices):
        if axis == 0:
            slice_data = grid[idx, :, :]
        elif axis == 1:
            slice_data = grid[:, idx, :]
        else:
            slice_data = grid[:, :, idx]
        im = ax.imshow(
            slice_data.T,
            origin="lower",
            cmap="RdBu_r",
            vmin=-abs_max,
            vmax=abs_max,
            interpolation="bilinear",
        )
        # Draw the zero-level isoline (molecular surface)
        ax.contour(slice_data.T, levels=[0.0], colors="k", linewidths=1.5)
        ax.set_title(f"{axis_labels[axis]}={idx}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = f"SDF slices — {molecule_id}" if molecule_id else "SDF slices"
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[plot_slices] saved → {out}")
    plt.close(fig)
    return {"num_slices": num_slices, "shape": list(grid.shape)}
