from __future__ import annotations


def build_loss(name: str):
    """Historical tiny per-loss builder kept for older tests/scripts."""
    if name != "weak_prior":
        raise ValueError(f"unsupported legacy loss: {name}")

    def _legacy_weak_prior(predictions: dict[str, object], targets: dict[str, object]) -> float:
        sdf = float(predictions.get("sdf", 0.0))
        values = targets.get("values", [])
        target = float(values[0]) if values else 0.0
        return abs(sdf - target)

    return _legacy_weak_prior

