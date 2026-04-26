from __future__ import annotations

import pytest


pytest.importorskip("torch")

from biomol_surface_unsup.training.loss_scheduler import LossWeightScheduler
from biomol_surface_unsup.training.trainer import Trainer


def test_best_model_metric_switches_to_vism_after_staged_ramp() -> None:
    scheduler = LossWeightScheduler(
        initial_weights={"init_sdf": 1.0, "area": 0.0},
        final_weights={"init_sdf": 0.0, "area": 1.0},
        warmup_epochs=0,
        mode="staged",
        pretrain_epochs=5,
        ramp_epochs=10,
    )
    trainer = object.__new__(Trainer)
    trainer.best_model_physical_start_epoch = Trainer._physical_best_start_epoch(scheduler)

    assert trainer.best_model_physical_start_epoch == 15
    assert trainer._best_model_metric_name(14) == "total"
    assert trainer._best_model_metric_name(15) == "vism_objective"


def test_best_model_metric_value_uses_vism_fallbacks() -> None:
    value, name = Trainer._best_model_metric_value(
        {"vism_total": 0.25, "total": 1.0},
        "vism_objective",
    )

    assert value == pytest.approx(0.25)
    assert name == "vism_total"
