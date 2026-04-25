from __future__ import annotations

from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover - environment fallback
    torch = None
    DataLoader = None

from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.losses.loss_builder import build_loss_fn
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.training.checkpoint import load_checkpoint, save_checkpoint
from biomol_surface_unsup.training.loss_scheduler import LossWeightScheduler
from biomol_surface_unsup.training.optimizer import build_optimizer
from biomol_surface_unsup.training.train_step import train_step
from biomol_surface_unsup.utils.config import normalize_loss_config


class Trainer:
    def __init__(self, cfg):
        if torch is None or DataLoader is None:
            raise RuntimeError("torch is required to run Trainer in this environment")

        self.cfg = cfg
        requested_device = str(cfg["train"].get("device", "cpu"))
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0

        data_cfg = cfg["data"]
        train_cfg = cfg["train"]
        self.log_every = int(train_cfg.get("log_every", 10))
        self.save_every = int(train_cfg.get("save_every", 0))
        self.output_dir = Path(train_cfg.get("output_dir", "outputs/checkpoints"))
        self.resume_from = train_cfg.get("resume_from")
        self.grad_clip_norm = train_cfg.get("grad_clip_norm")
        self.adaptive_surface_sampling = bool(
            data_cfg.get("adaptive_surface_sampling", train_cfg.get("adaptive_surface_sampling", False))
        )
        self.adaptive_surface_oversample = int(
            data_cfg.get("adaptive_surface_oversample", train_cfg.get("adaptive_surface_oversample", 8))
        )
        self.adaptive_surface_candidate_chunk_size = int(
            data_cfg.get(
                "adaptive_surface_candidate_chunk_size",
                train_cfg.get("adaptive_surface_candidate_chunk_size", 4096),
            )
        )
        self.area_importance_sampling = bool(
            data_cfg.get("area_importance_sampling", train_cfg.get("area_importance_sampling", False))
        )
        self.area_importance_band_width = float(
            data_cfg.get("area_importance_band_width", train_cfg.get("area_importance_band_width", 0.25))
        )
        self.area_importance_oversample = int(
            data_cfg.get("area_importance_oversample", train_cfg.get("area_importance_oversample", 32))
        )
        self.area_importance_candidate_chunk_size = int(
            data_cfg.get(
                "area_importance_candidate_chunk_size",
                train_cfg.get("area_importance_candidate_chunk_size", 4096),
            )
        )
        batch_size = int(train_cfg.get("batch_size", 1))
        raw_num_samples = data_cfg.get("num_samples")
        dataset_num_samples = int(raw_num_samples) if raw_num_samples is not None else None
        bbox_padding = float(data_cfg.get("bbox_padding", data_cfg.get("loose_surface_padding", 2.0)))
        self.train_dataset = MoleculeDataset(
            root=data_cfg.get("root", "data/processed"),
            split=data_cfg.get("train_split", "train"),
            num_samples=dataset_num_samples,
            num_query_points=int(data_cfg.get("num_query_points", 32)),
            bbox_padding=bbox_padding,
            initialization_mode=str(data_cfg.get("initialization_mode", "tight_atomic")),
            loose_surface_padding=float(data_cfg.get("loose_surface_padding", bbox_padding)),
            domain_padding=float(data_cfg.get("domain_padding", data_cfg.get("bbox_padding", bbox_padding))),
            containment_jitter=float(data_cfg.get("containment_jitter", 0.15)), # 包裹损失
            surface_band_width=float(
                data_cfg.get("surface_band_width", data_cfg.get("surface_bandwidth", 0.25))
            ),
            num_area_points=int(data_cfg.get("num_area_points", data_cfg.get("area_num_points", 0))),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=bool(train_cfg.get("shuffle", True)),
            num_workers=int(train_cfg.get("num_workers", 0)),
            collate_fn=collate_fn,
        )

        self.model = SurfaceModel.from_config(cfg.get("model", {}), num_atom_types=self.train_dataset.num_atom_types).to(
            self.device
        )
        if self.device == "cuda" and self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        loss_runtime_cfg = dict(cfg)
        loss_runtime_cfg["loss"] = normalize_loss_config(cfg.get("loss", {}))
        self.loss_fn = build_loss_fn(loss_runtime_cfg) # 如何组装的
        loss_cfg = loss_runtime_cfg["loss"]
        anneal_cfg = dict(loss_cfg.get("anneal", {}))
        initial_weights = anneal_cfg.get("initial_weights")
        final_weights = anneal_cfg.get("final_weights")
        initial_groups = anneal_cfg.get("initial_groups")
        final_groups = anneal_cfg.get("final_groups")
        self.loss_weight_scheduler = None
        if initial_weights is not None and final_weights is not None:
            self.loss_weight_scheduler = LossWeightScheduler(
                initial_weights=initial_weights,
                final_weights=final_weights,
                warmup_epochs=int(anneal_cfg.get("warmup_epochs", 0)),
                initial_groups=initial_groups,
                final_groups=final_groups,
                mode=str(anneal_cfg.get("schedule", anneal_cfg.get("weight_schedule", "linear"))),
                pretrain_epochs=int(anneal_cfg.get("pretrain_epochs", anneal_cfg.get("init_epochs", 0))),
                ramp_epochs=anneal_cfg.get("ramp_epochs"),
            )
        # Pressure annealing: step-function schedule for the pressure parameter.
        self.pressure_schedule = None
        if "initial_pressure" in anneal_cfg:
            self.pressure_schedule = {
                "initial": float(anneal_cfg["initial_pressure"]),
                "final": float(loss_cfg.get("pressure", 0.01)),
                "warmup_epochs": int(anneal_cfg.get("pressure_warmup_epochs",
                                                     anneal_cfg.get("warmup_epochs", 0))),
            }
        self.optimizer = build_optimizer(
            self.model,
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
        )
        self.start_epoch = 0
        self.global_step = 0
        self.last_metrics = None
        if self.resume_from:
            checkpoint = load_checkpoint(
                self.resume_from,
                self.model,
                optimizer=self.optimizer,
                map_location=self.device,
            )
            self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
            self.global_step = int(checkpoint.get("step", 0))
            self.last_metrics = checkpoint.get("metrics", {})

    def _checkpoint_path_for_epoch(self, epoch: int) -> Path:
        return self.output_dir / f"epoch_{epoch:04d}.pt"

    def _save_checkpoint(self, epoch: int, step: int, metrics: dict[str, float]) -> None:
        epoch_path = self._checkpoint_path_for_epoch(epoch)
        latest_path = self.output_dir / "latest.pt"
        save_checkpoint(
            epoch_path,
            self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
        )
        save_checkpoint(
            latest_path,
            self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

    @staticmethod
    def _batch_debug_summary(batch: dict) -> dict[str, object]:
        atom_counts = batch["atom_mask"].sum(dim=1).tolist()
        query_counts = batch["query_mask"].sum(dim=1).tolist()
        containment_counts = batch["containment_mask"].sum(dim=1).tolist()
        return {
            "ids": batch["id"],
            "atom_counts": [int(count) for count in atom_counts],
            "query_counts": [int(count) for count in query_counts],
            "containment_counts": [int(count) for count in containment_counts],
        }

    def _device_memory_summary(self) -> dict[str, float]:
        if self.device != "cuda" or torch is None or not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_gb": round(torch.cuda.memory_allocated() / (1024 ** 3), 3),
            "cuda_reserved_gb": round(torch.cuda.memory_reserved() / (1024 ** 3), 3),
            "cuda_max_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
            "cuda_max_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3),
        }

    @staticmethod
    def _loss_debug_summary(metrics: dict[str, float]) -> dict[str, float]:
        total = float(metrics.get("total", 0.0))
        summary = {}
        for name in ("vism_nonpolar", "electrostatic", "vism_total"):
            if name in metrics:
                value = float(metrics[name])
                summary[name] = round(value, 6)
                if abs(total) > 1e-12:
                    summary[f"{name}_ratio"] = round(value / total, 6)
        for name in ("area", "tolman_curvature", "pressure_volume", "lj_body"):
            if name in metrics:
                summary[name] = round(float(metrics[name]), 6)
        if "init_sdf" in metrics:
            summary["init_sdf"] = round(float(metrics["init_sdf"]), 6)
        for name in ("vism_total_energy", "vism_total_density"):
            if name in metrics:
                summary[name] = round(float(metrics[name]), 6)
        for name in (
            "sdf_abs_mean",
            "sdf_abs_max",
            "raw_residual_abs_mean",
            "raw_residual_abs_max",
            "sdf_minus_base_abs_mean",
            "delta_band_count",
            "area_delta_band_count",
            "area_query_count",
            "area_importance_band_count",
            "area_importance_candidate_count",
            "area_importance_hit_count",
            "area_importance_hit_rate",
            "area_importance_volume",
            "area_importance_volume_mean",
            "area_importance_phi_abs_mean",
            "area_importance_phi_abs_max",
            "area_importance_replacement_count",
            "area_importance_fallback_count",
            "lj_body_delta_band_count",
            "electrostatic_delta_band_count",
            "adaptive_surface_band_count",
            "adaptive_surface_candidate_count",
            "adaptive_surface_phi_abs_mean",
            "adaptive_surface_phi_abs_max",
            "sampling_area",
        ):
            if name in metrics:
                summary[name] = round(float(metrics[name]), 6)
        return summary

    def train(self):
        num_epochs = int(self.cfg["train"].get("epochs", 1))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float('inf')

        for epoch in range(self.start_epoch, num_epochs):
            loss_weights = None if self.loss_weight_scheduler is None else self.loss_weight_scheduler.get_weights(epoch)
            loss_group_overrides = None
            if self.loss_weight_scheduler is not None:
                group_overrides = self.loss_weight_scheduler.get_groups(epoch)
                loss_group_overrides = group_overrides or None
            if loss_weights is not None or loss_group_overrides is not None:
                print(
                    f"[trainer] epoch={epoch} "
                    f"loss_weights={loss_weights if loss_weights is not None else {}} "
                    f"loss_group_overrides={loss_group_overrides if loss_group_overrides is not None else {}}"
                )
            # Pressure annealing
            pressure_override = None
            if self.pressure_schedule is not None:
                if epoch < self.pressure_schedule["warmup_epochs"]:
                    pressure_override = self.pressure_schedule["initial"]
                else:
                    pressure_override = self.pressure_schedule["final"]
                print(f"[trainer] epoch={epoch} pressure={pressure_override}")
            latest_metrics = None
            epoch_total_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(self.train_loader):
                batch_summary = self._batch_debug_summary(batch)
                try:
                    metrics = train_step(
                        self.model,
                        batch,
                        self.loss_fn,
                        self.optimizer,
                        self.device,
                        loss_weights=loss_weights,
                        loss_group_overrides=loss_group_overrides,
                        grad_clip_norm=self.grad_clip_norm,
                        adaptive_surface_sampling=self.adaptive_surface_sampling,
                        adaptive_surface_oversample=self.adaptive_surface_oversample,
                        adaptive_surface_candidate_chunk_size=self.adaptive_surface_candidate_chunk_size,
                        area_importance_sampling=self.area_importance_sampling,
                        area_importance_band_width=self.area_importance_band_width,
                        area_importance_oversample=self.area_importance_oversample,
                        area_importance_candidate_chunk_size=self.area_importance_candidate_chunk_size,
                        pressure_override=pressure_override,
                    )
                except RuntimeError as exc:
                    print(
                        f"[!] Training failed at epoch={epoch} step={step} "
                        f"batch={batch_summary} error={exc}"
                    )
                    raise
                self.global_step += 1
                latest_metrics = metrics
                
                epoch_total_loss += metrics.get("total", 0.0)
                num_batches += 1
                
                if step % self.log_every == 0:
                    memory_summary = self._device_memory_summary()
                    loss_summary = self._loss_debug_summary(metrics)
                    print(
                        f"epoch={epoch} step={step} batch={batch_summary} "
                        f"memory={memory_summary} loss_summary={loss_summary} metrics={metrics}"
                    )

            if latest_metrics is not None:
                self.last_metrics = latest_metrics
                
                if num_batches > 0:
                    avg_loss = epoch_total_loss / num_batches
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        best_path = self.output_dir / "best_model.pt"
                        save_checkpoint(
                            best_path,
                            self.model,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            step=self.global_step,
                            metrics=latest_metrics,
                        )
                        print(f"[*] Saved new best model at epoch {epoch} with avg loss: {best_loss:.4f}")

                if self.save_every > 0 and ((epoch + 1) % self.save_every == 0 or epoch == num_epochs - 1):
                    self._save_checkpoint(epoch=epoch, step=self.global_step, metrics=latest_metrics)

    def evaluate(self):
        print("TODO")
