from __future__ import annotations

import torch

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
)
from biomol_surface_unsup.losses.area import _safe_query_grads, area_loss, mean_curvature_integral_fd
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.eikonal import eikonal_loss
from biomol_surface_unsup.losses.electrostatics import electrostatic_free_energy_cfa
from biomol_surface_unsup.losses.vdw import lj_body_integral
from biomol_surface_unsup.losses.pressure_volume import pressure_volume_loss
from biomol_surface_unsup.losses.weak_prior import weak_prior_loss
from biomol_surface_unsup.legacy.losses import build_loss as _legacy_build_loss
from biomol_surface_unsup.utils.pairwise import chunked_smooth_atomic_union_field
from biomol_surface_unsup.utils.config import normalize_loss_config


QUERY_GROUP_IDS = {
    "global": QUERY_GROUP_GLOBAL,
    "containment": QUERY_GROUP_CONTAINMENT,
    "surface_band": QUERY_GROUP_SURFACE_BAND,
}

SUPPORTED_LOSSES = (
    "containment",
    "weak_prior",
    "area",
    "tolman_curvature",
    "pressure_volume",
    "lj_body",
    "electrostatic",
    "eikonal",
)

VISM_COMPONENT_LOSSES = (
    "area",
    "tolman_curvature",
    "pressure_volume",
    "lj_body",
    "electrostatic",
)


def _batched_atomic_union_field(coords: torch.Tensor, radii: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    return chunked_smooth_atomic_union_field(coords, radii, query_points)


def _masked_count(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return mask.sum().to(dtype)


def _group_mask(query_group: torch.Tensor, query_mask: torch.Tensor, group_names: list[str]) -> torch.Tensor:
    mask = torch.zeros_like(query_group, dtype=torch.bool)
    for group_name in group_names:
        if group_name not in QUERY_GROUP_IDS:
            supported = ", ".join(sorted(QUERY_GROUP_IDS))
            raise ValueError(f"unsupported query group '{group_name}', expected one of: {supported}")
        mask = mask | (query_group == QUERY_GROUP_IDS[group_name])
    return mask & query_mask


def _domain_volume_from_batch(batch: dict[str, torch.Tensor], reference: torch.Tensor) -> torch.Tensor | None:
    bbox_volume = batch.get("bbox_volume")
    if bbox_volume is not None:
        return torch.as_tensor(bbox_volume, dtype=reference.dtype, device=reference.device).reshape(-1)

    bbox_lower = batch.get("bbox_lower")
    bbox_upper = batch.get("bbox_upper")
    if bbox_lower is None or bbox_upper is None:
        return None

    lower = torch.as_tensor(bbox_lower, dtype=reference.dtype, device=reference.device)
    upper = torch.as_tensor(bbox_upper, dtype=reference.dtype, device=reference.device)
    return (upper - lower).prod(dim=-1).reshape(-1)


def _normalize_vism_objective(loss_cfg: dict[str, object]) -> str:
    objective = str(loss_cfg.get("vism_objective", "energy")).lower()
    normalization = loss_cfg.get("vism_normalization")
    if normalization is not None:
        normalization_name = str(normalization).lower()
        if normalization_name in {"bbox_volume", "volume", "energy_density", "density"}:
            objective = "energy_density"
        elif normalization_name in {"none", "energy"}:
            objective = "energy"
        else:
            raise ValueError(
                "unsupported vism_normalization, expected one of: none, energy, bbox_volume, volume, density"
            )

    if objective in {"energy", "free_energy", "total_energy"}:
        return "energy"
    if objective in {"energy_density", "density", "bbox_volume", "volume_normalized"}:
        return "energy_density"
    raise ValueError("unsupported vism_objective, expected 'energy' or 'energy_density'")


def build_loss_fn(cfg: dict[str, object]):
    loss_cfg = normalize_loss_config(dict(cfg.get("loss", {})))
    configured_losses = loss_cfg["losses"]
    vism_objective = _normalize_vism_objective(loss_cfg)
    delta_eps = float(loss_cfg.get("delta_eps", 0.1))
    heaviside_eps = float(loss_cfg.get("heaviside_eps", 0.1))
    containment_margin = float(loss_cfg.get("containment_margin", 0.5))
    pressure = float(loss_cfg.get("pressure", 0.01))
    rho_0 = float(loss_cfg.get("rho_0", 0.0334))
    gamma_0 = float(loss_cfg.get("gamma_0", loss_cfg.get("surface_tension", 0.1315)))
    tolman_length = float(loss_cfg.get("tolman_length", loss_cfg.get("tau", 1.0)))
    tolman_fd_offset = float(loss_cfg.get("tolman_fd_offset", delta_eps))
    eps_solvent = float(loss_cfg.get("eps_solvent", 78.0))
    eps_solute = float(loss_cfg.get("eps_solute", 1.0))
    electrostatic_dist_eps = float(loss_cfg.get("electrostatic_dist_eps", 1.0))
    weak_prior_target = str(loss_cfg.get("weak_prior_target", "atomic_union")).lower()

    def effective_weight(loss_name: str, loss_weights: dict[str, float] | None) -> float:
        if loss_weights is not None and loss_name in loss_weights:
            return float(loss_weights[loss_name])
        return float(configured_losses[loss_name]["weight"])

    def loss_fn(
        batch: dict[str, torch.Tensor],
        model_out: dict[str, torch.Tensor],
        loss_weights: dict[str, float] | None = None,
        loss_group_overrides: dict[str, list[str]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute and aggregate all configured unsupervised physical losses.
        
        Input Shapes:
        - batch["coords"]: [B, N, 3] (3D coordinates of all atoms)
        - batch["radii"]: [B, N] (Van der Waals radii per atom)
        - batch["epsilon"] / ["sigma"]: [B, N] (optional LJ energy parameters)
        - batch["atom_mask"]: [B, N] (Boolean mask for valid padded atoms)
        - batch["query_points"]: [B, Q, 3] (3D spatial probes)
        - batch["query_group"]: [B, Q] (Integer IDs denoting the source/type of each probe)
        - batch["query_mask"]: [B, Q] (Boolean mask for valid padded probes)
        
        - model_out["sdf"]: [B, Q] (Predicted SDF distances corresponding to query_points)
        
        Output Shape:
        - returns: dict[str, torch.Tensor] 
                   Mapping of string loss names to single 0-dimensional scalar Tensors (shape: []).
                   Includes individual component losses, debugging counts, and the final combined "total".
        """
        coords = batch["coords"]  # [B, N, 3]
        radii = batch["radii"]  # [B, N]
        charges = batch.get("charges")
        epsilon = batch.get("epsilon")
        sigma = batch.get("sigma")
        if charges is None:
            charges = radii.new_zeros(radii.shape)
        if epsilon is None:
            epsilon = radii.new_zeros(radii.shape)
        if sigma is None:
            sigma = radii.new_zeros(radii.shape)
        atom_mask = batch["atom_mask"]  # [B, N]
        query_points = batch["query_points"]  # [B, Q, 3]
        query_group = batch["query_group"]  # [B, Q]
        query_mask = batch["query_mask"]  # [B, Q]
        pred_sdf = model_out["sdf"]  # [B, Q]
        surface_bbox_lower = batch.get("surface_bbox_lower")
        surface_bbox_upper = batch.get("surface_bbox_upper")
        if surface_bbox_lower is None:
            surface_bbox_lower = batch.get("bbox_lower")
        if surface_bbox_upper is None:
            surface_bbox_upper = batch.get("bbox_upper")

        if pred_sdf.ndim == 1:
            pred_sdf = pred_sdf.unsqueeze(0)
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
            query_group = query_group.unsqueeze(0)
            query_mask = query_mask.unsqueeze(0)
            coords = coords.unsqueeze(0)
            radii = radii.unsqueeze(0)
            charges = charges.unsqueeze(0)
            epsilon = epsilon.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            atom_mask = atom_mask.unsqueeze(0)
        bbox_volume = _domain_volume_from_batch(batch, pred_sdf)
        if bbox_volume is not None and bbox_volume.numel() == 1 and pred_sdf.shape[0] > 1:
            bbox_volume = bbox_volume.expand(pred_sdf.shape[0])
        if not query_points.requires_grad:
            query_points = query_points.requires_grad_(True)

        base_masks = {
            "global": _group_mask(query_group, query_mask, ["global"]),
            "containment": _group_mask(query_group, query_mask, ["containment"]),
            "surface_band": _group_mask(query_group, query_mask, ["surface_band"]),
        }
        active_groups = {}
        for loss_name in SUPPORTED_LOSSES:
            groups = configured_losses[loss_name]["groups"]
            if loss_group_overrides is not None and loss_name in loss_group_overrides:
                groups = loss_group_overrides[loss_name]
            active_groups[loss_name] = list(groups)
        loss_masks = {
            loss_name: _group_mask(query_group, query_mask, active_groups[loss_name])
            for loss_name in SUPPORTED_LOSSES
        }
        if bbox_volume is None and vism_objective == "energy_density":
            raise ValueError(
                "vism_objective='energy_density' requires bbox_volume. "
                "Use biomol_surface_unsup.datasets.collate.collate_fn, or pass bbox_volume/bbox_lower/bbox_upper "
                "through the training batch."
            )
        if (
            bbox_volume is None
            and effective_weight("tolman_curvature", loss_weights) != 0.0
            and torch.any(loss_masks["tolman_curvature"])
        ):
            raise ValueError(
                "tolman_curvature requires bbox_volume for physical Monte Carlo normalization. "
                "Use biomol_surface_unsup.datasets.collate.collate_fn, or pass bbox_volume/bbox_lower/bbox_upper "
                "through the training batch."
            )

        query_grads = _safe_query_grads(pred_sdf, query_points)
        batch_size = pred_sdf.shape[0]
        component_weights = {
            loss_name: effective_weight(loss_name, loss_weights)
            for loss_name in VISM_COMPONENT_LOSSES
        }
        zero_per_sample = pred_sdf.new_zeros((batch_size,))
        area_energy = zero_per_sample
        if component_weights["area"] != 0.0:
            area_energy = area_loss(
                pred_sdf,
                query_points,
                mask=loss_masks["area"],
                eps=delta_eps,
                query_grads=query_grads,
                domain_volume=bbox_volume,
                reduction="none",
            ) * pred_sdf.new_tensor(gamma_0)
        tolman_energy = pred_sdf.new_zeros((batch_size,))
        if bbox_volume is not None and component_weights["tolman_curvature"] != 0.0:
            tolman_energy = mean_curvature_integral_fd(
                pred_sdf,
                query_points,
                mask=loss_masks["tolman_curvature"],
                eps=delta_eps,
                offset=tolman_fd_offset,
                query_grads=query_grads,
                domain_volume=bbox_volume,
                reduction="none",
            ) * pred_sdf.new_tensor(-2.0 * gamma_0 * tolman_length)
        pressure_energy = zero_per_sample
        if component_weights["pressure_volume"] != 0.0:
            pressure_energy = pressure_volume_loss(
                pred_sdf,
                mask=loss_masks["pressure_volume"],
                pressure=pressure,
                eps=heaviside_eps,
                domain_volume=bbox_volume,
                reduction="none",
            )
        lj_energy = zero_per_sample
        if component_weights["lj_body"] != 0.0:
            lj_energy = lj_body_integral(
                pred_sdf=pred_sdf,
                query_points=query_points,
                coords=coords,
                epsilon_lj=epsilon,
                sigma_lj=sigma,
                atom_mask=atom_mask,
                mask=loss_masks["lj_body"],
                rho_0=rho_0,
                eps_h=heaviside_eps,
                domain_volume=bbox_volume,
                reduction="none",
            )
        electrostatic_energy = zero_per_sample
        if component_weights["electrostatic"] != 0.0:
            electrostatic_energy = electrostatic_free_energy_cfa(
                pred_sdf=pred_sdf,
                query_points=query_points,
                coords=coords,
                charges=charges,
                atom_mask=atom_mask,
                mask=loss_masks["electrostatic"],
                eps_solvent=eps_solvent,
                eps_solute=eps_solute,
                eps_h=heaviside_eps,
                dist_eps=electrostatic_dist_eps,
                domain_volume=bbox_volume,
                reduction="none",
            )
        component_energy = {
            "area": area_energy,
            "tolman_curvature": tolman_energy,
            "pressure_volume": pressure_energy,
            "lj_body": lj_energy,
            "electrostatic": electrostatic_energy,
        }
        if bbox_volume is None:
            component_density = component_energy
        else:
            safe_volume = bbox_volume.to(dtype=pred_sdf.dtype, device=pred_sdf.device).clamp_min(1e-12)
            component_density = {
                name: value / safe_volume
                for name, value in component_energy.items()
            }
        selected_components = component_density if vism_objective == "energy_density" else component_energy

        losses = {
            "weak_prior": weak_prior_loss(
                coords,
                radii,
                query_points,
                pred_sdf,
                mask=loss_masks["weak_prior"],
                atom_mask=atom_mask,
                target_type=weak_prior_target,
                bbox_lower=surface_bbox_lower,
                bbox_upper=surface_bbox_upper,
            ),
            "eikonal": eikonal_loss(
                pred_sdf,
                query_points,
                mask=loss_masks["eikonal"],
                query_grads=query_grads,
            ),
            "containment": containment_loss(
                pred_sdf,
                margin=containment_margin,
                mask=loss_masks["containment"],
            ),
        }
        for loss_name in VISM_COMPONENT_LOSSES:
            losses[loss_name] = selected_components[loss_name].mean()
            losses[f"{loss_name}_energy"] = component_energy[loss_name].mean()
            if bbox_volume is not None:
                losses[f"{loss_name}_density"] = component_density[loss_name].mean()

        vism_nonpolar_energy = area_energy + tolman_energy + pressure_energy + lj_energy
        vism_total_energy = vism_nonpolar_energy + electrostatic_energy
        if bbox_volume is None:
            vism_nonpolar_density = vism_nonpolar_energy
            vism_total_density = vism_total_energy
        else:
            vism_nonpolar_density = vism_nonpolar_energy / safe_volume
            vism_total_density = vism_total_energy / safe_volume
        selected_nonpolar = vism_nonpolar_density if vism_objective == "energy_density" else vism_nonpolar_energy
        selected_total = vism_total_density if vism_objective == "energy_density" else vism_total_energy
        losses["vism_nonpolar"] = selected_nonpolar.mean()
        losses["vism_total"] = selected_total.mean()
        losses["vism_nonpolar_energy"] = vism_nonpolar_energy.mean()
        losses["vism_total_energy"] = vism_total_energy.mean()
        losses["vism_energy"] = losses["vism_total_energy"]
        if bbox_volume is not None:
            losses["vism_nonpolar_density"] = vism_nonpolar_density.mean()
            losses["vism_total_density"] = vism_total_density.mean()
            losses["vism_density"] = losses["vism_total_density"]
        losses["vism_objective"] = losses["vism_total"]
        safe_coords = coords.masked_fill(~atom_mask.unsqueeze(-1), 0.0)
        safe_radii = radii.masked_fill(~atom_mask, 0.0)
        with torch.no_grad():
            target_sdf = _batched_atomic_union_field(safe_coords, safe_radii, query_points)
        losses["target_sdf"] = target_sdf[query_mask].mean() if torch.any(query_mask) else pred_sdf.new_zeros(())
        losses["global_count"] = _masked_count(base_masks["global"], pred_sdf.dtype)
        losses["containment_count"] = _masked_count(base_masks["containment"], pred_sdf.dtype)
        losses["surface_band_count"] = _masked_count(base_masks["surface_band"], pred_sdf.dtype)
        for loss_name in SUPPORTED_LOSSES:
            losses[f"{loss_name}_count"] = _masked_count(loss_masks[loss_name], pred_sdf.dtype)

        total = pred_sdf.new_zeros(())
        for loss_name in SUPPORTED_LOSSES:
            weight = effective_weight(loss_name, loss_weights)
            total = total + weight * losses[loss_name]
        losses["total"] = total
        return losses

    return loss_fn


def build_loss(name: str):
    """Compatibility shim forwarding the old helper into the legacy module."""
    return _legacy_build_loss(name)
