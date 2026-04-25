from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
)
from biomol_surface_unsup.datasets.collate import collate_fn
from biomol_surface_unsup.datasets.molecule_dataset import MoleculeDataset
from biomol_surface_unsup.losses.containment import containment_loss
from biomol_surface_unsup.losses.eikonal import eikonal_loss
from biomol_surface_unsup.losses.lj_body import lj_body_integral
from biomol_surface_unsup.losses.pressure_volume import pressure_volume_loss
from biomol_surface_unsup.losses.area import area_loss
from biomol_surface_unsup.losses.loss_builder import build_loss, build_loss_fn
from biomol_surface_unsup.training.loss_scheduler import LossWeightScheduler
from biomol_surface_unsup.training.train_step import train_step
from biomol_surface_unsup.utils.config import normalize_loss_config
from biomol_surface_unsup.models.surface_model import SurfaceModel
from biomol_surface_unsup.geometry.sdf_ops import box_sdf


def test_build_loss_and_call() -> None:
    loss_fn = build_loss("weak_prior")
    value = loss_fn({"sdf": 1.0}, {"values": [0.0]})
    assert isinstance(value, float)
    assert value == pytest.approx(1.0)


def test_containment_loss_uses_margin_penalty() -> None:
    pred_sdf = torch.tensor([[-0.8, -0.6, 0.1]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True]])
    value = containment_loss(pred_sdf, margin=0.5, mask=mask)
    expected = torch.tensor([0.0, 0.0, 0.36], dtype=torch.float32).mean()
    assert torch.allclose(value, expected)


def _build_batch() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    query_points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.0, 1.0, 0.5], [0.2, 0.1, 0.0]],
            [[0.5, 0.0, 0.0], [0.2, 0.3, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_sdf = query_points.pow(2).sum(dim=-1) - 0.2
    batch = {
        "coords": torch.tensor(
            [
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 1.1, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "atom_types": torch.tensor([[1, 6, 8], [1, 6, 0]], dtype=torch.long),
        "radii": torch.tensor([[1.2, 1.3, 1.1], [1.1, 1.0, 0.0]], dtype=torch.float32),
        "epsilon": torch.tensor([[0.2, 0.3, 0.4], [0.2, 0.25, 0.0]], dtype=torch.float32),
        "sigma": torch.tensor([[2.0, 2.1, 2.2], [1.9, 2.0, 0.0]], dtype=torch.float32),
        "atom_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "query_points": query_points,
        "query_group": torch.tensor(
            [
                [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_SURFACE_BAND, QUERY_GROUP_CONTAINMENT],
                [QUERY_GROUP_GLOBAL, QUERY_GROUP_SURFACE_BAND, QUERY_GROUP_GLOBAL, QUERY_GROUP_GLOBAL],
            ],
            dtype=torch.long,
        ),
        "query_mask": torch.tensor(
            [[True, True, True, True], [True, True, False, False]],
            dtype=torch.bool,
        ),
        "containment_points": torch.zeros((2, 2, 3), dtype=torch.float32),
        "containment_mask": torch.tensor([[True, True], [False, False]], dtype=torch.bool),
    }
    return batch, pred_sdf


def test_normalize_loss_config_preserves_default_behavior() -> None:
    normalized = normalize_loss_config(
        {
            "lambda_area": 1.0,
            "lambda_volume": 0.5,
            "lambda_containment": 2.0,
            "lambda_prior": 0.5,
            "lambda_eikonal": 0.5,
        }
    )

    assert normalized["losses"]["containment"] == {"weight": 2.0, "groups": ["containment"]}
    assert normalized["losses"]["weak_prior"] == {"weight": 0.5, "groups": ["surface_band"]}
    assert normalized["losses"]["area"] == {"weight": 1.0, "groups": ["surface_band"]}
    assert normalized["losses"]["pressure_volume"] == {"weight": 0.5, "groups": ["global"]}
    assert normalized["losses"]["lj_body"] == {"weight": 0.0, "groups": ["global"]}
    assert normalized["losses"]["eikonal"] == {"weight": 0.5, "groups": ["global", "surface_band"]}


def test_pressure_volume_loss_matches_smoothed_exterior_fraction() -> None:
    pred_sdf = torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True, False]])

    value = pressure_volume_loss(pred_sdf, mask=mask, pressure=0.2, eps=0.1)
    expected = 0.2 * torch.tensor([0.0317255, 0.5], dtype=torch.float32).mean()

    assert value.item() == pytest.approx(expected.item(), rel=1e-4)


def test_lj_body_integral_matches_single_atom_closed_form() -> None:
    pred_sdf = torch.tensor([[1.0]], dtype=torch.float32)
    query_points = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
    coords = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
    epsilon = torch.tensor([[0.5]], dtype=torch.float32)
    sigma = torch.tensor([[1.0]], dtype=torch.float32)
    atom_mask = torch.tensor([[True]])
    mask = torch.tensor([[True]])

    value = lj_body_integral(
        pred_sdf,
        query_points,
        coords,
        epsilon,
        sigma,
        atom_mask,
        mask=mask,
        rho_0=1.0,
        eps_h=0.1,
        dist_eps=1e-6,
    )
    expected_lj = 4.0 * 0.5 * ((1.0 / 2.0) ** 12 - (1.0 / 2.0) ** 6)
    expected = expected_lj * pressure_volume_loss(torch.tensor([[1.0]]), pressure=1.0, eps=0.1).item()
    assert value.item() == pytest.approx(expected, rel=1e-5)


def test_loss_weight_scheduler_linearly_interpolates() -> None:
    scheduler = LossWeightScheduler(
        initial_weights={"weak_prior": 1.0, "area": 0.0},
        final_weights={"weak_prior": 0.0, "area": 1.0},
        warmup_epochs=4,
    )
    assert scheduler.get_weights(0) == pytest.approx({"area": 0.0, "weak_prior": 1.0})
    assert scheduler.get_weights(2) == pytest.approx({"area": 0.5, "weak_prior": 0.5})
    assert scheduler.get_weights(4) == pytest.approx({"area": 1.0, "weak_prior": 0.0})


def test_loss_weight_scheduler_step_mode_switches_after_warmup() -> None:
    scheduler = LossWeightScheduler(
        initial_weights={"init_sdf": 1.0, "area": 0.0},
        final_weights={"init_sdf": 0.0, "area": 1.0},
        warmup_epochs=5,
        mode="step",
    )

    assert scheduler.get_weights(0) == pytest.approx({"area": 0.0, "init_sdf": 1.0})
    assert scheduler.get_weights(4) == pytest.approx({"area": 0.0, "init_sdf": 1.0})
    assert scheduler.get_weights(5) == pytest.approx({"area": 1.0, "init_sdf": 0.0})


def test_area_and_eikonal_backward_remain_finite_at_zero_query_gradients() -> None:
    query_points = torch.zeros((1, 4, 3), dtype=torch.float32, requires_grad=True)
    pred_sdf = (query_points * 0.0).sum(dim=-1)
    mask = torch.ones((1, 4), dtype=torch.bool)

    total = area_loss(pred_sdf, query_points, mask=mask) + eikonal_loss(pred_sdf, query_points, mask=mask)
    total.backward()

    assert torch.isfinite(query_points.grad).all()


def test_area_and_eikonal_backward_remain_finite_with_query_on_atom_center() -> None:
    model = SurfaceModel(num_atom_types=16)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=torch.float32)
    atom_types = torch.tensor([1, 6, 8], dtype=torch.long)
    radii = torch.tensor([1.2, 1.3, 1.1], dtype=torch.float32)
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.2, 0.1, 0.0], [0.0, 1.0, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )
    pred_sdf = model(coords, atom_types, radii, query_points)["sdf"]
    mask = torch.ones_like(pred_sdf, dtype=torch.bool)

    total = area_loss(pred_sdf, query_points, mask=mask) + eikonal_loss(pred_sdf, query_points, mask=mask)
    total.backward()

    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_build_loss_fn_returns_weighted_losses_from_default_mapping() -> None:
    batch, pred_sdf = _build_batch()
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 2.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.5, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "pressure_volume": {"weight": 0.5, "groups": ["global"]},
                    "lj_body": {"weight": 0.25, "groups": ["global"]},
                    "eikonal": {"weight": 0.5, "groups": ["global", "surface_band"]},
                },
                "containment_margin": 0.5,
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert {"area", "pressure_volume", "lj_body", "containment", "weak_prior", "eikonal", "total"}.issubset(losses)
    assert "volume" not in losses
    assert "volume_count" not in losses
    assert losses["containment_count"].item() == 2.0
    assert losses["global_count"].item() == 2.0
    assert losses["surface_band_count"].item() == 2.0
    assert losses["weak_prior_count"].item() == 2.0
    assert losses["area_count"].item() == 2.0
    assert losses["pressure_volume_count"].item() == 2.0
    assert losses["lj_body_count"].item() == 2.0
    assert losses["eikonal_count"].item() == 4.0
    assert losses["containment"].ndim == 0
    assert losses["total"].ndim == 0
    assert float(losses["total"].detach().cpu()) >= 0.0


def test_build_loss_fn_supports_multi_group_union_masks() -> None:
    batch, pred_sdf = _build_batch()
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 1.0, "groups": ["containment", "surface_band"]},
                    "weak_prior": {"weight": 1.0, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "pressure_volume": {"weight": 1.0, "groups": ["global"]},
                    "lj_body": {"weight": 1.0, "groups": ["global"]},
                    "eikonal": {"weight": 1.0, "groups": ["global", "surface_band"]},
                }
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert losses["containment_count"].item() == pytest.approx(4.0)
    assert losses["surface_band_count"].item() == pytest.approx(2.0)
    assert losses["eikonal_count"].item() == pytest.approx(4.0)
    assert losses["containment"].item() >= 0.0
    assert losses["total"].ndim == 0


def test_build_loss_fn_handles_empty_masks_from_configured_groups() -> None:
    batch, pred_sdf = _build_batch()
    batch["query_group"] = torch.tensor(
        [
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_CONTAINMENT, QUERY_GROUP_GLOBAL],
            [QUERY_GROUP_GLOBAL, QUERY_GROUP_GLOBAL, QUERY_GROUP_GLOBAL, QUERY_GROUP_GLOBAL],
        ],
        dtype=torch.long,
    )
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 1.0, "groups": []},
                    "weak_prior": {"weight": 1.0, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "pressure_volume": {"weight": 1.0, "groups": []},
                    "lj_body": {"weight": 1.0, "groups": []},
                    "eikonal": {"weight": 1.0, "groups": []},
                }
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    assert losses["area"].item() == pytest.approx(0.0)
    assert losses["weak_prior"].item() == pytest.approx(0.0)
    assert losses["area_count"].item() == pytest.approx(0.0)
    assert losses["weak_prior_count"].item() == pytest.approx(0.0)
    assert losses["containment"].item() == pytest.approx(0.0)
    assert losses["containment_count"].item() == pytest.approx(0.0)
    assert losses["eikonal"].item() == pytest.approx(0.0)
    assert losses["eikonal_count"].item() == pytest.approx(0.0)
    assert losses["pressure_volume"].item() == pytest.approx(0.0)
    assert losses["pressure_volume_count"].item() == pytest.approx(0.0)
    assert losses["lj_body"].item() == pytest.approx(0.0)
    assert losses["lj_body_count"].item() == pytest.approx(0.0)
    assert "volume_count" not in losses


def test_energy_density_objective_normalizes_vism_terms_per_sample() -> None:
    batch, pred_sdf = _build_batch()
    batch["bbox_volume"] = torch.tensor([10.0, 100.0], dtype=torch.float32)
    loss_fn = build_loss_fn(
        {
            "loss": {
                "vism_objective": "energy_density",
                "losses": {
                    "containment": {"weight": 0.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.0, "groups": ["surface_band"]},
                    "area": {"weight": 0.0, "groups": ["global"]},
                    "tolman_curvature": {"weight": 0.0, "groups": ["global"]},
                    "pressure_volume": {"weight": 0.0, "groups": ["global"]},
                    "lj_body": {"weight": 1.0, "groups": ["global"]},
                    "electrostatic": {"weight": 0.0, "groups": ["global"]},
                    "eikonal": {"weight": 0.0, "groups": ["global", "surface_band"]},
                },
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    global_mask = (batch["query_group"] == QUERY_GROUP_GLOBAL) & batch["query_mask"]
    per_sample_lj_energy = lj_body_integral(
        pred_sdf,
        batch["query_points"],
        batch["coords"],
        batch["epsilon"],
        batch["sigma"],
        batch["atom_mask"],
        mask=global_mask,
        rho_0=0.0334,
        eps_h=0.1,
        domain_volume=batch["bbox_volume"],
        reduction="none",
    )
    expected_density = (per_sample_lj_energy / batch["bbox_volume"]).mean()

    assert losses["lj_body"].item() == pytest.approx(expected_density.item(), rel=1e-5)
    assert losses["lj_body_energy"].item() == pytest.approx(per_sample_lj_energy.mean().item(), rel=1e-5)
    assert losses["lj_body_density"].item() == pytest.approx(expected_density.item(), rel=1e-5)
    assert losses["total"].item() == pytest.approx(expected_density.item(), rel=1e-5)


def test_energy_objective_uses_raw_integrated_vism_terms() -> None:
    batch, pred_sdf = _build_batch()
    batch["bbox_volume"] = torch.tensor([10.0, 100.0], dtype=torch.float32)
    loss_fn = build_loss_fn(
        {
            "loss": {
                "vism_objective": "energy",
                "losses": {
                    "containment": {"weight": 0.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.0, "groups": ["surface_band"]},
                    "init_sdf": {"weight": 0.0, "groups": ["global"]},
                    "area": {"weight": 0.0, "groups": ["global"]},
                    "tolman_curvature": {"weight": 0.0, "groups": ["global"]},
                    "pressure_volume": {"weight": 0.0, "groups": ["global"]},
                    "lj_body": {"weight": 1.0, "groups": ["global"]},
                    "electrostatic": {"weight": 0.0, "groups": ["global"]},
                    "eikonal": {"weight": 0.0, "groups": ["global", "surface_band"]},
                },
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})
    global_mask = (batch["query_group"] == QUERY_GROUP_GLOBAL) & batch["query_mask"]
    per_sample_lj_energy = lj_body_integral(
        pred_sdf,
        batch["query_points"],
        batch["coords"],
        batch["epsilon"],
        batch["sigma"],
        batch["atom_mask"],
        mask=global_mask,
        rho_0=0.0334,
        eps_h=0.1,
        domain_volume=batch["bbox_volume"],
        reduction="none",
    )

    assert losses["lj_body"].item() == pytest.approx(per_sample_lj_energy.mean().item(), rel=1e-5)
    assert losses["lj_body_energy"].item() == pytest.approx(per_sample_lj_energy.mean().item(), rel=1e-5)
    assert losses["total"].item() == pytest.approx(per_sample_lj_energy.mean().item(), rel=1e-5)


def test_init_sdf_loss_fits_box_sdf_with_mse() -> None:
    batch, _ = _build_batch()
    batch["surface_bbox_lower"] = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0]], dtype=torch.float32)
    batch["surface_bbox_upper"] = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=torch.float32)
    target = box_sdf(batch["query_points"], batch["surface_bbox_lower"], batch["surface_bbox_upper"])
    pred_sdf = target + 0.5
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "init_sdf": {"weight": 1.0, "groups": ["global", "containment", "surface_band"]},
                    "containment": {"weight": 0.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.0, "groups": ["surface_band"]},
                    "area": {"weight": 0.0, "groups": ["global"]},
                    "tolman_curvature": {"weight": 0.0, "groups": ["global"]},
                    "pressure_volume": {"weight": 0.0, "groups": ["global"]},
                    "lj_body": {"weight": 0.0, "groups": ["global"]},
                    "electrostatic": {"weight": 0.0, "groups": ["global"]},
                    "eikonal": {"weight": 0.0, "groups": ["global"]},
                },
            }
        }
    )

    losses = loss_fn(batch, {"sdf": pred_sdf})

    assert losses["init_sdf"].item() == pytest.approx(0.25)
    assert losses["total"].item() == pytest.approx(0.25)


def test_energy_density_objective_requires_bbox_volume() -> None:
    batch, pred_sdf = _build_batch()
    loss_fn = build_loss_fn({"loss": {"vism_objective": "energy_density"}})

    with pytest.raises(ValueError, match="energy_density"):
        loss_fn(batch, {"sdf": pred_sdf})


def test_train_step_runs_backward_and_optimizer_step_on_batched_toy_batch() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, coords, atom_types, radii, query_points, atom_mask=None, query_mask=None):
            del coords, atom_types, radii, atom_mask
            sdf = self.scale * query_points.pow(2).sum(dim=-1) + self.bias
            if query_mask is not None:
                sdf = sdf * query_mask.to(sdf.dtype)
            return {"sdf": sdf}

    batch, _ = _build_batch()
    loss_fn = build_loss_fn({"loss": {}})
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    before = model.scale.detach().clone()
    metrics = train_step(model, batch, loss_fn, optimizer, device="cpu")
    after = model.scale.detach().clone()

    assert "total" in metrics
    assert metrics["total"] >= 0.0
    assert "volume_count" not in metrics
    assert metrics["containment_count"] == pytest.approx(2.0)
    assert not torch.allclose(before, after)


def test_train_step_reports_grad_norm_when_clipping_enabled() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(5.0, dtype=torch.float32))

        def forward(self, coords, atom_types, radii, query_points, atom_mask=None, query_mask=None):
            del coords, atom_types, radii, atom_mask
            sdf = self.scale * query_points.pow(2).sum(dim=-1)
            if query_mask is not None:
                sdf = sdf * query_mask.to(sdf.dtype)
            return {"sdf": sdf}

    batch, _ = _build_batch()
    loss_fn = build_loss_fn({"loss": {}})
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics = train_step(model, batch, loss_fn, optimizer, device="cpu", grad_clip_norm=0.1)

    assert "grad_norm" in metrics
    assert metrics["grad_norm"] >= 0.0


def test_train_step_adaptive_surface_sampling_reports_selected_band() -> None:
    class PlaneModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        def forward(self, coords, atom_types, radii, query_points, atom_mask=None, query_mask=None):
            del coords, atom_types, radii, atom_mask
            sdf = query_points[..., 0] + self.bias
            if query_mask is not None:
                sdf = sdf * query_mask.to(sdf.dtype)
            return {"sdf": sdf}

    torch.manual_seed(0)
    batch, _ = _build_batch()
    batch["domain_bbox_lower"] = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], dtype=torch.float32)
    batch["domain_bbox_upper"] = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    batch["bbox_lower"] = batch["domain_bbox_lower"]
    batch["bbox_upper"] = batch["domain_bbox_upper"]
    batch["bbox_volume"] = torch.tensor([8.0, 8.0], dtype=torch.float32)
    loss_fn = build_loss_fn(
        {
            "loss": {
                "losses": {
                    "containment": {"weight": 0.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.0, "groups": ["surface_band"]},
                    "area": {"weight": 1.0, "groups": ["surface_band"]},
                    "tolman_curvature": {"weight": 0.0, "groups": ["surface_band"]},
                    "pressure_volume": {"weight": 0.0, "groups": ["global"]},
                    "lj_body": {"weight": 0.0, "groups": ["global"]},
                    "electrostatic": {"weight": 0.0, "groups": ["global"]},
                    "eikonal": {"weight": 0.0, "groups": ["global", "surface_band"]},
                }
            }
        }
    )
    model = PlaneModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics = train_step(
        model,
        batch,
        loss_fn,
        optimizer,
        device="cpu",
        adaptive_surface_sampling=True,
        adaptive_surface_oversample=32,
        adaptive_surface_candidate_chunk_size=16,
    )

    assert metrics["adaptive_surface_band_count"] == pytest.approx(2.0)
    assert metrics["adaptive_surface_candidate_count"] == pytest.approx(66.0)
    assert metrics["adaptive_surface_phi_abs_mean"] >= 0.0
    assert metrics["adaptive_surface_phi_abs_max"] < 0.25


def test_loose_box_single_batch_forward_loss_is_finite() -> None:
    torch.manual_seed(0)
    dataset = MoleculeDataset(
        num_samples=1,
        num_query_points=16,
        initialization_mode="loose_box",
        loose_surface_padding=4.0,
        domain_padding=8.0,
    )
    batch = collate_fn([dataset[0]])
    query_points = batch["query_points"].requires_grad_(True)
    model = SurfaceModel(
        num_atom_types=dataset.num_atom_types,
        local_hidden_dim=32,
        local_out_dim=32,
        global_hidden_dim=32,
        global_out_dim=32,
        decoder_hidden_dim=32,
        sdf_base_type="box",
        zero_init_output=True,
    )
    out = model(
        batch["coords"],
        batch["atom_types"],
        batch["radii"],
        query_points,
        charges=batch["charges"],
        epsilon=batch["epsilon"],
        sigma=batch["sigma"],
        atom_mask=batch["atom_mask"],
        query_mask=batch["query_mask"],
        bbox_lower=batch["surface_bbox_lower"],
        bbox_upper=batch["surface_bbox_upper"],
    )
    loss_fn = build_loss_fn(
        {
            "loss": {
                "weak_prior_target": "box",
                "losses": {
                    "containment": {"weight": 0.0, "groups": ["containment"]},
                    "weak_prior": {"weight": 0.2, "groups": ["surface_band"]},
                    "area": {"weight": 0.0, "groups": ["surface_band"]},
                    "tolman_curvature": {"weight": 0.0, "groups": ["surface_band"]},
                    "pressure_volume": {"weight": 0.0, "groups": ["global"]},
                    "lj_body": {"weight": 0.0, "groups": ["global"]},
                    "electrostatic": {"weight": 0.0, "groups": ["global"]},
                    "eikonal": {"weight": 0.1, "groups": ["global", "surface_band"]},
                },
            }
        }
    )
    losses = loss_fn(
        {
            "coords": batch["coords"],
            "atom_types": batch["atom_types"],
            "radii": batch["radii"],
            "charges": batch["charges"],
            "epsilon": batch["epsilon"],
            "sigma": batch["sigma"],
            "atom_mask": batch["atom_mask"],
            "query_points": query_points,
            "query_group": batch["query_group"],
            "query_mask": batch["query_mask"],
            "containment_points": batch["containment_points"],
            "containment_mask": batch["containment_mask"],
            "surface_bbox_lower": batch["surface_bbox_lower"],
            "surface_bbox_upper": batch["surface_bbox_upper"],
            "domain_bbox_lower": batch["domain_bbox_lower"],
            "domain_bbox_upper": batch["domain_bbox_upper"],
            "bbox_lower": batch["bbox_lower"],
            "bbox_upper": batch["bbox_upper"],
            "bbox_volume": batch["bbox_volume"],
        },
        out,
    )

    assert torch.isfinite(losses["total"])
    assert losses["global_count"].item() > 0.0
    assert losses["surface_band_count"].item() > 0.0
