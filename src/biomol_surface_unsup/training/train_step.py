import inspect

import torch

from biomol_surface_unsup.datasets.sampling import QUERY_GROUP_SURFACE_BAND


def _has_nonfinite_gradients(model) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return True
    return False


def _model_accepts_physics_inputs(model) -> bool:
    parameters = inspect.signature(model.forward).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    names = {param.name for param in parameters}
    return {"charges", "epsilon", "sigma"}.issubset(names)


def _model_accepts_return_aux(model) -> bool:
    parameters = inspect.signature(model.forward).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    return "return_aux" in {param.name for param in parameters}


def _model_accepts_bbox_inputs(model) -> bool:
    parameters = inspect.signature(model.forward).parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return True
    names = {param.name for param in parameters}
    return {"bbox_lower", "bbox_upper"}.issubset(names)


def _optional_tensor_to_device(batch, key, device):
    value = batch.get(key)
    if value is None:
        return None
    return value.to(device)


def _sample_adaptive_surface_band(
    model,
    coords,
    atom_types,
    radii,
    query_points,
    query_group,
    query_mask,
    *,
    charges=None,
    epsilon=None,
    sigma=None,
    atom_mask=None,
    bbox_lower=None,
    bbox_upper=None,
    domain_bbox_lower=None,
    domain_bbox_upper=None,
    oversample=8,
    candidate_chunk_size=4096,
) -> tuple[torch.Tensor, dict[str, float]]:
    surface_mask = (query_group == QUERY_GROUP_SURFACE_BAND) & query_mask
    surface_counts = surface_mask.sum(dim=1)
    max_surface_count = int(surface_counts.max().item()) if surface_counts.numel() > 0 else 0
    if max_surface_count == 0:
        return query_points, {}

    lower = domain_bbox_lower if domain_bbox_lower is not None else bbox_lower
    upper = domain_bbox_upper if domain_bbox_upper is not None else bbox_upper
    if lower is None or upper is None:
        return query_points, {}

    batch_size = int(query_points.shape[0])
    oversample = max(int(oversample), 1)
    num_random = max_surface_count * oversample
    random_candidates = lower.unsqueeze(1) + torch.rand(
        batch_size,
        num_random,
        3,
        dtype=query_points.dtype,
        device=query_points.device,
    ) * (upper - lower).unsqueeze(1)
    random_mask = torch.ones(
        (batch_size, num_random),
        dtype=torch.bool,
        device=query_points.device,
    )

    existing_surface = query_points.new_zeros((batch_size, max_surface_count, 3))
    existing_mask = torch.zeros((batch_size, max_surface_count), dtype=torch.bool, device=query_points.device)
    for batch_idx in range(batch_size):
        count = int(surface_counts[batch_idx].item())
        if count == 0:
            continue
        existing_surface[batch_idx, :count] = query_points[batch_idx, surface_mask[batch_idx]]
        existing_mask[batch_idx, :count] = True

    candidates = torch.cat([random_candidates, existing_surface], dim=1)
    candidate_mask = torch.cat([random_mask, existing_mask], dim=1)
    candidate_abs_sdf = query_points.new_empty(candidates.shape[:2])

    model_kwargs = {"atom_mask": atom_mask}
    if _model_accepts_physics_inputs(model):
        model_kwargs.update({"charges": charges, "epsilon": epsilon, "sigma": sigma})
    if _model_accepts_return_aux(model):
        model_kwargs["return_aux"] = False
    if _model_accepts_bbox_inputs(model):
        model_kwargs.update({"bbox_lower": bbox_lower, "bbox_upper": bbox_upper})

    candidate_chunk_size = max(int(candidate_chunk_size), 1)
    was_training = bool(model.training)
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, candidates.shape[1], candidate_chunk_size):
                end = min(start + candidate_chunk_size, candidates.shape[1])
                chunk_mask = candidate_mask[:, start:end]
                out = model(
                    coords,
                    atom_types,
                    radii,
                    candidates[:, start:end],
                    query_mask=chunk_mask,
                    **model_kwargs,
                )
                candidate_abs_sdf[:, start:end] = out["sdf"].detach().abs()
    finally:
        if was_training:
            model.train()
    candidate_abs_sdf = candidate_abs_sdf.masked_fill(~candidate_mask, float("inf"))
    selected_abs_sdf, selected_index = torch.topk(
        candidate_abs_sdf,
        k=max_surface_count,
        dim=1,
        largest=False,
    )

    adapted_query_points = query_points.clone()
    selected_values = []
    for batch_idx in range(batch_size):
        count = int(surface_counts[batch_idx].item())
        if count == 0:
            continue
        replacement = candidates[batch_idx, selected_index[batch_idx, :count]]
        adapted_query_points[batch_idx, surface_mask[batch_idx]] = replacement
        selected_values.append(selected_abs_sdf[batch_idx, :count])

    if selected_values:
        selected_abs = torch.cat(selected_values)
        metrics = {
            "adaptive_surface_band_count": float(selected_abs.numel()),
            "adaptive_surface_candidate_count": float(candidate_mask.sum().item()),
            "adaptive_surface_phi_abs_mean": float(selected_abs.mean().cpu()),
            "adaptive_surface_phi_abs_max": float(selected_abs.max().cpu()),
        }
    else:
        metrics = {}
    return adapted_query_points, metrics


def train_step(
    model,
    batch,
    loss_fn,
    optimizer,
    device,
    loss_weights=None,
    loss_group_overrides=None,
    grad_clip_norm=None,
    adaptive_surface_sampling=False,
    adaptive_surface_oversample=8,
    adaptive_surface_candidate_chunk_size=4096,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    coords = batch["coords"].to(device)
    atom_types = batch["atom_types"].to(device)
    radii = batch["radii"].to(device)
    charges = batch["charges"].to(device) if "charges" in batch else None
    epsilon = batch["epsilon"].to(device) if "epsilon" in batch else None
    sigma = batch["sigma"].to(device) if "sigma" in batch else None
    res_ids = batch["res_ids"].to(device) if "res_ids" in batch else None
    atom_mask = batch["atom_mask"].to(device)
    query_points = batch["query_points"].to(device)  # [B, Q, 3]
    query_group = batch["query_group"].to(device)  # [B, Q]
    query_mask = batch["query_mask"].to(device)  # [B, Q]
    containment_points = batch["containment_points"].to(device)  # [B, C, 3]
    containment_mask = batch["containment_mask"].to(device)  # [B, C]
    bbox_lower = _optional_tensor_to_device(batch, "bbox_lower", device)
    bbox_upper = _optional_tensor_to_device(batch, "bbox_upper", device)
    bbox_volume = _optional_tensor_to_device(batch, "bbox_volume", device)
    surface_bbox_lower = _optional_tensor_to_device(batch, "surface_bbox_lower", device)
    surface_bbox_upper = _optional_tensor_to_device(batch, "surface_bbox_upper", device)
    domain_bbox_lower = _optional_tensor_to_device(batch, "domain_bbox_lower", device)
    domain_bbox_upper = _optional_tensor_to_device(batch, "domain_bbox_upper", device)
    adaptive_metrics = {}
    if adaptive_surface_sampling:
        query_points, adaptive_metrics = _sample_adaptive_surface_band(
            model,
            coords,
            atom_types,
            radii,
            query_points,
            query_group,
            query_mask,
            charges=charges,
            epsilon=epsilon,
            sigma=sigma,
            atom_mask=atom_mask,
            bbox_lower=surface_bbox_lower if surface_bbox_lower is not None else bbox_lower,
            bbox_upper=surface_bbox_upper if surface_bbox_upper is not None else bbox_upper,
            domain_bbox_lower=domain_bbox_lower,
            domain_bbox_upper=domain_bbox_upper,
            oversample=adaptive_surface_oversample,
            candidate_chunk_size=adaptive_surface_candidate_chunk_size,
        )
    query_points = query_points.requires_grad_(True)

    model_kwargs = {
        "atom_mask": atom_mask,
        "query_mask": query_mask,
    }
    if _model_accepts_physics_inputs(model):
        model_kwargs.update(
            {
                "charges": charges,
                "epsilon": epsilon,
                "sigma": sigma,
            }
        )

    if _model_accepts_return_aux(model):
        model_kwargs["return_aux"] = False
    if _model_accepts_bbox_inputs(model):
        model_kwargs.update(
            {
                "bbox_lower": surface_bbox_lower if surface_bbox_lower is not None else bbox_lower,
                "bbox_upper": surface_bbox_upper if surface_bbox_upper is not None else bbox_upper,
            }
        )

    out = model(coords, atom_types, radii, query_points, **model_kwargs)
    losses = loss_fn(
        {
            "coords": coords,
            "atom_types": atom_types,
            "radii": radii,
            "charges": charges,
            "epsilon": epsilon,
            "sigma": sigma,
            "res_ids": res_ids,
            "atom_mask": atom_mask,
            "query_points": query_points,
            "query_group": query_group,
            "query_mask": query_mask,
            "containment_points": containment_points,
            "containment_mask": containment_mask,
            "surface_bbox_lower": surface_bbox_lower,
            "surface_bbox_upper": surface_bbox_upper,
            "domain_bbox_lower": domain_bbox_lower,
            "domain_bbox_upper": domain_bbox_upper,
            "bbox_lower": bbox_lower,
            "bbox_upper": bbox_upper,
            "bbox_volume": bbox_volume,
        },
        out,
        loss_weights=loss_weights,
        loss_group_overrides=loss_group_overrides,
    )
    if not torch.isfinite(losses["total"]):
        raise ValueError(f"non-finite total loss before backward: {float(losses['total'].detach().cpu())}")
    losses["total"].backward()
    grad_norm = None
    if grad_clip_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("non-finite gradient norm encountered during clipping")
    if _has_nonfinite_gradients(model):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("non-finite gradients encountered before optimizer step")
    optimizer.step()

    metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
    metrics.update(adaptive_metrics)
    raw_residual = out.get("raw_residual")
    base_phi = out.get("base_phi")
    if raw_residual is not None:
        residual_detached = raw_residual.detach()
        metrics["raw_residual_mean"] = float(residual_detached.mean().cpu())
        metrics["raw_residual_abs_mean"] = float(residual_detached.abs().mean().cpu())
        metrics["raw_residual_abs_max"] = float(residual_detached.abs().max().cpu())
    if base_phi is not None:
        base_delta = (out["sdf"] - base_phi).detach()
        metrics["sdf_minus_base_abs_mean"] = float(base_delta.abs().mean().cpu())
        metrics["sdf_minus_base_abs_max"] = float(base_delta.abs().max().cpu())
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm.detach().cpu())
    metrics.update({f"sampling_{k}": float(v) for k, v in batch.get("sampling_counts", {}).items()})

    del out, losses
    return metrics
