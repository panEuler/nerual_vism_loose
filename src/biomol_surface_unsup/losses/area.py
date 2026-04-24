from __future__ import annotations

import torch


def smooth_delta(phi: torch.Tensor, eps: float) -> torch.Tensor:
    eps = max(float(eps), 1e-6)
    x = phi / eps
    return torch.where(
        x.abs() <= 1.0,
        0.5 / eps * (1.0 + torch.cos(torch.pi * x)),
        torch.zeros_like(phi),
    )


def _safe_query_grads(pred_sdf: torch.Tensor, query_points: torch.Tensor) -> torch.Tensor:
    """Return d(pred_sdf.sum()) / d(query_points) with batched support.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - query_points: [Q, 3] or [B, Q, 3]
    - return: same leading shape as query_points
    """
    grads = torch.autograd.grad(
        outputs=pred_sdf,
        inputs=query_points,
        grad_outputs=torch.ones_like(pred_sdf),
        create_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]
    if grads is None:
        grads = torch.zeros_like(query_points)
    return grads


def _stable_grad_norm(grads: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable gradient norm used inside higher-order losses."""
    return torch.sqrt(grads.pow(2).sum(dim=-1) + float(eps))


def _masked_monte_carlo_integral(
    integrand: torch.Tensor,
    domain_volume: torch.Tensor | None = None,
    mask: torch.Tensor | None = None, # [B, Q]
    reduction: str = "mean",
) -> torch.Tensor:
    """带掩码的蒙特卡洛体积积分。
    
    利用随机采样点求近似积分：积分值 ≈ 样本均值 * 积分域体积。
    该实现支持批处理 (Batched)，并能够通过掩码 (Mask) 自动忽略无效的填充采样点。
    """
    # 1. 维度对齐：如果输入是单样本 [Q] 而非批次 [B, Q]，则人为增加 Batch 维度，统一化处理
    if integrand.ndim == 1:
        integrand = integrand.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)

    # 2. 掩码初始化：如果没有提供掩码，则默认所有采样点都是有效的
    if mask is None:
        mask = torch.ones_like(integrand, dtype=torch.bool)
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be either 'mean' or 'none'")
    # 如果全图没有任何有效点，安全统一返回 0，防止后续报错
    if not torch.any(mask):
        if reduction == "none":
            return integrand.new_zeros((integrand.shape[0],))
        return integrand.new_zeros(())

    # 3. 积分域体积（Domain Volume）初始化：
    # 如果没有给定体积常数，则默认体积为 1（相当于退化为了只求均值）
    if domain_volume is None:
        domain_volume = torch.ones((integrand.shape[0],), dtype=integrand.dtype, device=integrand.device)
    else:
        # 确保体积数据的类型和设备（GPU/CPU）与 integrand 保持一致，并展平为一维 [B]
        domain_volume = torch.as_tensor(domain_volume, dtype=integrand.dtype, device=integrand.device).reshape(-1)

    # 4. 蒙特卡洛积分核心计算：
    # 4.1 抹除无效点干扰：将 Batch 对齐产生的假点/无效点强制置 0
    masked_integrand = integrand * mask.to(integrand.dtype)
    # 4.2 计算每个 batch 中有效点的个数，限制最小值 1 防止求均值时除以 0 (导致 NaN)
    counts = mask.sum(dim=-1).clamp_min(1).to(integrand.dtype) # [B] 
    # 4.3 求有效样本的均值：用真实的总和除以真实的采样点数量
    per_sample_means = masked_integrand.sum(dim=-1) / counts # [B]
    # 4.4 蒙特卡洛公式：样本均值乘空间体积，得到每个样本真正的体积积分值
    per_sample_integrals = per_sample_means * domain_volume
    
    # 5. 批次级别 (Batch-level) 聚合：
    # 找出哪些 batch 样本里确实包含有效数据（即 > 0）
    valid_samples = (mask.sum(dim=-1) > 0).to(integrand.dtype) # [B] 
    if reduction == "none":
        return per_sample_integrals * valid_samples
    # 把批次里所有有效的计算结果加起来，求算术平均，作为整个 Batch 返回的终版 Loss 值
    return (per_sample_integrals * valid_samples).sum() / valid_samples.sum().clamp_min(1.0)


def _mean_curvature_from_level_set(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    query_grads: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    grad_norm = _stable_grad_norm(grads)
    unit_normals = grads / grad_norm.unsqueeze(-1)

    divergence = torch.zeros_like(pred_sdf)
    for axis in range(query_points.shape[-1]):
        component = unit_normals[..., axis]
        component_grad = torch.autograd.grad(
            outputs=component,
            inputs=query_points,
            grad_outputs=torch.ones_like(component),
            create_graph=True,
            only_inputs=True,
            allow_unused=True,
        )[0]
        if component_grad is None:
            continue
        divergence = divergence + component_grad[..., axis]
    mean_curvature = 0.5 * divergence
    return mean_curvature, grad_norm


def area_loss(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    query_grads: torch.Tensor | None = None,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Toy surface-area surrogate on masked batched queries.

    Shapes:
    - pred_sdf: [Q] or [B, Q]
    - query_points: [Q, 3] or [B, Q, 3]
    - mask: [Q] or [B, Q] or None
    """
    grads = _safe_query_grads(pred_sdf, query_points) if query_grads is None else query_grads
    integrand = smooth_delta(pred_sdf, eps) * _stable_grad_norm(grads)

    if domain_volume is None:
        if reduction == "none":
            return _masked_monte_carlo_integral(
                integrand,
                domain_volume=None,
                mask=mask,
                reduction=reduction,
            )
        if mask is not None and not torch.any(mask):
            return pred_sdf.new_zeros(())
        if mask is not None:
            integrand = integrand[mask]
        return integrand.mean()
    return _masked_monte_carlo_integral(
        integrand,
        domain_volume=domain_volume,
        mask=mask,
        reduction=reduction,
    )


def mean_curvature_integral(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    query_grads: torch.Tensor | None = None,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if domain_volume is None:
        raise ValueError("mean_curvature_integral requires domain_volume for physical Monte Carlo normalization")
    mean_curvature, grad_norm = _mean_curvature_from_level_set(
        pred_sdf,
        query_points,
        query_grads=query_grads,
    )
    integrand = smooth_delta(pred_sdf, eps) * grad_norm * mean_curvature
    return _masked_monte_carlo_integral(
        integrand,
        domain_volume=domain_volume,
        mask=mask,
        reduction=reduction,
    )


def mean_curvature_integral_fd(
    pred_sdf: torch.Tensor,
    query_points: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 0.1,
    offset: float | None = None,
    query_grads: torch.Tensor | None = None,
    domain_volume: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Approximate ``∫_Γ H dS`` via a symmetric finite difference of offset areas.

    For a signed-distance-like field, the area of the level set ``phi = s`` satisfies:

        A(s) = A(0) + 2 s ∫_Γ H dS + O(s^2)

    which gives the first-order approximation

        ∫_Γ H dS ≈ (A(+h) - A(-h)) / (4 h)

    with ``A(+h)`` implemented by evaluating the area surrogate on ``pred_sdf - h``
    and ``A(-h)`` on ``pred_sdf + h``. This avoids the expensive second-order
    autodiff used by the exact divergence-of-normals formulation and only relies
    on first-order query gradients already needed for the area term.
    """
    if domain_volume is None:
        raise ValueError("mean_curvature_integral_fd requires domain_volume for physical Monte Carlo normalization")

    fd_offset = max(float(offset if offset is not None else eps), 1e-4)
    area_outer = area_loss(
        pred_sdf - fd_offset,
        query_points,
        mask=mask,
        eps=eps,
        query_grads=query_grads,
        domain_volume=domain_volume,
        reduction=reduction,
    )
    area_inner = area_loss(
        pred_sdf + fd_offset,
        query_points,
        mask=mask,
        eps=eps,
        query_grads=query_grads,
        domain_volume=domain_volume,
        reduction=reduction,
    )
    return (area_outer - area_inner) / (4.0 * fd_offset)
