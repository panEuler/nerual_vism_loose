import inspect

import torch


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


def train_step(
    model,
    batch,
    loss_fn,
    optimizer,
    device,
    loss_weights=None,
    loss_group_overrides=None,
    grad_clip_norm=None,
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
    query_points = batch["query_points"].to(device).requires_grad_(True)  # [B, Q, 3]
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
    if grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm.detach().cpu())
    metrics.update({f"sampling_{k}": float(v) for k, v in batch.get("sampling_counts", {}).items()})

    del out, losses
    return metrics
