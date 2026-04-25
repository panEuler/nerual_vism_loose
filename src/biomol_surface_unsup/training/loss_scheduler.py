from __future__ import annotations


class LossWeightScheduler:
    def __init__(
        self,
        initial_weights: dict[str, float],
        final_weights: dict[str, float],
        warmup_epochs: int,
        initial_groups: dict[str, list[str]] | None = None,
        final_groups: dict[str, list[str]] | None = None,
        mode: str = "linear",
        pretrain_epochs: int = 0,
        ramp_epochs: int | None = None,
    ) -> None:
        self.initial_weights = {key: float(value) for key, value in initial_weights.items()}
        self.final_weights = {key: float(value) for key, value in final_weights.items()}
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.pretrain_epochs = max(int(pretrain_epochs), 0)
        self.ramp_epochs = self.warmup_epochs if ramp_epochs is None else max(int(ramp_epochs), 0)
        self.loss_names = sorted(set(self.initial_weights) | set(self.final_weights))
        self.initial_groups = {key: list(value) for key, value in (initial_groups or {}).items()}
        self.final_groups = {key: list(value) for key, value in (final_groups or {}).items()}
        self.mode = str(mode).lower()
        if self.mode not in {"linear", "step", "staged"}:
            raise ValueError("LossWeightScheduler mode must be one of: 'linear', 'step', 'staged'")

    def _interpolate_weights(self, alpha: float) -> dict[str, float]:
        alpha = min(max(float(alpha), 0.0), 1.0)
        return {
            name: (1.0 - alpha) * self.initial_weights.get(name, 0.0)
            + alpha * self.final_weights.get(name, self.initial_weights.get(name, 0.0))
            for name in self.loss_names
        }

    def get_weights(self, epoch: int) -> dict[str, float]:
        epoch = max(int(epoch), 0)
        if self.mode == "step":
            source = self.initial_weights if epoch < self.warmup_epochs else self.final_weights
            fallback = self.final_weights if source is self.initial_weights else self.initial_weights
            return {name: source.get(name, fallback.get(name, 0.0)) for name in self.loss_names}
        if self.mode == "staged":
            if epoch < self.pretrain_epochs:
                return {name: self.initial_weights.get(name, self.final_weights.get(name, 0.0)) for name in self.loss_names}
            if self.ramp_epochs == 0:
                return {name: self.final_weights.get(name, self.initial_weights.get(name, 0.0)) for name in self.loss_names}
            return self._interpolate_weights((epoch - self.pretrain_epochs + 1) / float(self.ramp_epochs))
        if self.warmup_epochs == 0:
            return {name: self.final_weights.get(name, self.initial_weights.get(name, 0.0)) for name in self.loss_names}
        return self._interpolate_weights(min(epoch, self.warmup_epochs) / float(self.warmup_epochs))

    def get_groups(self, epoch: int) -> dict[str, list[str]]:
        if not self.initial_groups and not self.final_groups:
            return {}
        epoch = max(int(epoch), 0)
        if self.mode == "staged":
            use_initial = epoch < self.pretrain_epochs + self.ramp_epochs
        elif self.warmup_epochs == 0:
            return {name: list(groups) for name, groups in self.final_groups.items()}
        else:
            use_initial = epoch < self.warmup_epochs
        if use_initial:
            source = self.initial_groups
        else:
            source = self.final_groups
        return {name: list(groups) for name, groups in source.items()}
