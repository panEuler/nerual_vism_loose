import inspect

import torch

from biomol_surface_unsup.datasets.sampling import QUERY_GROUP_AREA, QUERY_GROUP_SURFACE_BAND


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


def _model_forward_kwargs(
    model,
    *,
    charges=None,
    epsilon=None,
    sigma=None,
    atom_mask=None,
    bbox_lower=None,
    bbox_upper=None,
    return_aux: bool | None = None,
) -> dict:
    model_kwargs = {"atom_mask": atom_mask}
    if _model_accepts_physics_inputs(model):
        model_kwargs.update({"charges": charges, "epsilon": epsilon, "sigma": sigma})
    if return_aux is not None and _model_accepts_return_aux(model):
        model_kwargs["return_aux"] = return_aux
    if _model_accepts_bbox_inputs(model):
        model_kwargs.update({"bbox_lower": bbox_lower, "bbox_upper": bbox_upper})
    return model_kwargs


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

    model_kwargs = _model_forward_kwargs(
        model,
        charges=charges,
        epsilon=epsilon,
        sigma=sigma,
        atom_mask=atom_mask,
        bbox_lower=bbox_lower,
        bbox_upper=bbox_upper,
        return_aux=False,
    )

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


def _sample_area_importance_band(
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
    bbox_volume=None,
    band_width=0.25,
    oversample=32,
    candidate_chunk_size=4096,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    area_mask = (query_group == QUERY_GROUP_AREA) & query_mask
    area_counts = area_mask.sum(dim=1)
    max_area_count = int(area_counts.max().item()) if area_counts.numel() > 0 else 0
    if max_area_count == 0:
        return query_points, None, {}

    lower = domain_bbox_lower if domain_bbox_lower is not None else bbox_lower
    upper = domain_bbox_upper if domain_bbox_upper is not None else bbox_upper
    if lower is None or upper is None:
        return query_points, None, {}

    batch_size = int(query_points.shape[0])
    oversample = max(int(oversample), 1)
    num_random = max_area_count * oversample
    candidates = lower.unsqueeze(1) + torch.rand(
        batch_size,
        num_random,
        3,
        dtype=query_points.dtype,
        device=query_points.device,
    ) * (upper - lower).unsqueeze(1)
    candidate_abs_sdf = query_points.new_empty(candidates.shape[:2])

    model_kwargs = _model_forward_kwargs(
        model,
        charges=charges,
        epsilon=epsilon,
        sigma=sigma,
        atom_mask=atom_mask,
        bbox_lower=bbox_lower,
        bbox_upper=bbox_upper,
        return_aux=False,
    )

    candidate_chunk_size = max(int(candidate_chunk_size), 1)
    was_training = bool(model.training)
    if was_training:
        model.eval()
    try:
        with torch.no_grad():
            for start in range(0, candidates.shape[1], candidate_chunk_size):
                end = min(start + candidate_chunk_size, candidates.shape[1])
                chunk_mask = torch.ones(
                    (batch_size, end - start),
                    dtype=torch.bool,
                    device=query_points.device,
                )
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

    band_width = max(float(band_width), 1e-6)
    hit_mask = candidate_abs_sdf <= band_width
    hit_counts = hit_mask.sum(dim=1)
    if bbox_volume is None:
        volume = (upper - lower).prod(dim=-1)
    else:
        volume = bbox_volume.to(dtype=query_points.dtype, device=query_points.device).reshape(-1)
    area_importance_volume = volume.clone()

    adapted_query_points = query_points.clone()
    selected_values = []
    selected_total = 0
    replacement_total = 0
    fallback_total = 0
    for batch_idx in range(batch_size):
        count = int(area_counts[batch_idx].item())
        if count == 0:
            continue
        hit_index = torch.nonzero(hit_mask[batch_idx], as_tuple=False).flatten()
        if hit_index.numel() == 0:
            fallback_total += count
            continue

        area_importance_volume[batch_idx] = volume[batch_idx] * (
            hit_index.numel() / float(num_random)
        )
        if hit_index.numel() >= count:
            chosen = hit_index[torch.randperm(hit_index.numel(), device=query_points.device)[:count]]
        else:
            replacement_total += count - int(hit_index.numel())
            chosen = hit_index[torch.randint(0, hit_index.numel(), (count,), device=query_points.device)]
        adapted_query_points[batch_idx, area_mask[batch_idx]] = candidates[batch_idx, chosen]
        selected_values.append(candidate_abs_sdf[batch_idx, chosen])
        selected_total += count

    metrics = {
        "area_importance_band_count": float(selected_total),
        "area_importance_candidate_count": float(batch_size * num_random),
        "area_importance_hit_count": float(hit_counts.sum().item()),
        "area_importance_hit_rate": float(hit_counts.sum().item() / max(batch_size * num_random, 1)),
        "area_importance_volume_mean": float(area_importance_volume.mean().cpu()),
        "area_importance_replacement_count": float(replacement_total),
        "area_importance_fallback_count": float(fallback_total),
    }
    if selected_values:
        selected_abs = torch.cat(selected_values)
        metrics["area_importance_phi_abs_mean"] = float(selected_abs.mean().cpu())
        metrics["area_importance_phi_abs_max"] = float(selected_abs.max().cpu())

    return adapted_query_points, area_importance_volume, metrics


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
    area_importance_sampling=False,
    area_importance_band_width=0.25,
    area_importance_oversample=32,
    area_importance_candidate_chunk_size=4096,
    pressure_override=None,
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
    area_importance_volume = None
    if area_importance_sampling:
        query_points, area_importance_volume, area_metrics = _sample_area_importance_band(
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
            bbox_volume=bbox_volume,
            band_width=area_importance_band_width,
            oversample=area_importance_oversample,
            candidate_chunk_size=area_importance_candidate_chunk_size,
        )
        adaptive_metrics.update(area_metrics)
    query_points = query_points.requires_grad_(True)

    model_kwargs = _model_forward_kwargs(
        model,
        charges=charges,
        epsilon=epsilon,
        sigma=sigma,
        atom_mask=atom_mask,
        bbox_lower=surface_bbox_lower if surface_bbox_lower is not None else bbox_lower,
        bbox_upper=surface_bbox_upper if surface_bbox_upper is not None else bbox_upper,
        return_aux=False,
    )
    model_kwargs["query_mask"] = query_mask

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
            "area_importance_volume": area_importance_volume,
        },
        out,
        loss_weights=loss_weights,
        loss_group_overrides=loss_group_overrides,
        pressure_override=pressure_override,
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
    sdf_detached = out["sdf"].detach()
    metrics["sdf_mean"] = float(sdf_detached.mean().cpu())
    metrics["sdf_abs_mean"] = float(sdf_detached.abs().mean().cpu())
    metrics["sdf_abs_max"] = float(sdf_detached.abs().max().cpu())
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
