from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from biomol_surface_unsup.datasets.molecule_dataset import ATOM_TYPE_TO_ID
from biomol_surface_unsup.inference import load_processed_molecule, predict_sdf
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.training.checkpoint import load_checkpoint
from biomol_surface_unsup.utils.config import load_yaml


def _resolve_sample_dir(args: argparse.Namespace) -> Path:
    if args.processed_sample_dir:
        return Path(args.processed_sample_dir)

    if not args.pdb_file or not args.chain_id:
        raise ValueError("provide either --processed-sample-dir or both --pdb-file and --chain-id")

    from preprocess import process_one_pdb

    preprocess_dir = Path(args.preprocess_dir)
    process_one_pdb(args.pdb_file, args.chain_id, str(preprocess_dir))
    return preprocess_dir / Path(args.pdb_file).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--query-points", type=str, required=True, help="Path to a .npy file with shape [Q, 3].")
    parser.add_argument("--output", type=str, required=True, help="Path to save predicted SDF values as .npy.")
    parser.add_argument("--processed-sample-dir", type=str, default=None)
    parser.add_argument("--pdb-file", type=str, default=None)
    parser.add_argument("--chain-id", type=str, default=None)
    parser.add_argument("--preprocess-dir", type=str, default="outputs/infer_processed")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--loose-surface-padding",
        type=float,
        default=4.0,
        help="Padding used to reconstruct the loose box SDF base when the model config uses sdf_base.type=box.",
    )
    args = parser.parse_args()

    sample_dir = _resolve_sample_dir(args)
    molecule = load_processed_molecule(sample_dir)
    query_points = np.load(args.query_points)

    model_cfg = load_yaml(args.model_config)
    model = SurfaceModel.from_config(model_cfg, num_atom_types=len(ATOM_TYPE_TO_ID))
    load_checkpoint(args.checkpoint, model, map_location=args.device)

    sdf = predict_sdf(
        model=model,
        molecule=molecule,
        query_points=torch.as_tensor(query_points, dtype=torch.float32),
        device=args.device,
        chunk_size=args.chunk_size,
        loose_surface_padding=args.loose_surface_padding,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, sdf.numpy())
    print(f"saved {sdf.shape[0]} SDF values to {output_path}")


if __name__ == "__main__":
    main()
