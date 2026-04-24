# biomol_surface_unsup

Unsupervised neural implicit biomolecular surface learning with VISM-style objectives.

## Goals
- Input: molecule atoms
- Output: scalar implicit field / SDF
- Training: unsupervised or weakly anchored variational objective

## Current API

### Preprocess
- `python scripts/preprocess.py --config configs/data/toy.yaml`

### Train
- Small / toy debug:
  - `python scripts/train.py --config configs/experiment/debug.yaml`
- Main training:
  - `python scripts/train.py --config configs/experiment/my_train.yaml`
- Physical-objective compatibility alias:
  - `python scripts/train.py --config configs/experiment/my_train_vism_physical.yaml`

### Inference / mesh extraction
Predict a dense SDF grid and optionally extract a mesh:

```bash
# Full pipeline (SDF grid + OBJ mesh + PNG slice plots)
python scripts/infer_mesh.py \
  --ckpt outputs/checkpoints/latest.pt \
  --config configs/experiment/my_train.yaml \
  --split test \
  --spacing_angstrom 0.5 \
  --block_voxel_size 64 \
  --output_dir outputs/meshes

# Single processed sample directory
python scripts/infer_mesh.py \
  --ckpt outputs/checkpoints/latest.pt \
  --config configs/experiment/my_train.yaml \
  --processed_sample_dir data/test/1A5Z_ACBD \
  --spacing_angstrom 0.5 \
  --block_voxel_size 64 \
  --output_dir outputs/test \
  --no_mesh \
  --no_slices
```

Outputs per molecule:
- `<id>_sdf.npy`: raw SDF volume
- `<id>_sdf_meta.json`: grid origin / spacing / shape metadata
- `<id>_surface.obj`: extracted mesh when mesh export is enabled
- `<id>_slices.png`: SDF slice visualization when slice export is enabled

Optional dependencies:
- `pip install scikit-image` for marching cubes
- `pip install matplotlib` for slice plots

## Legacy API

Historical experiment templates and small compatibility helpers live under:

- `src/biomol_surface_unsup/legacy/`
- `configs/legacy/`

The active training/inference pipeline is now the physical VISM path described
above; the historical `vism_lite` pipeline has been removed.
