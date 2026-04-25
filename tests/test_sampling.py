from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.datasets.sampling import (
    QUERY_GROUP_AREA,
    QUERY_GROUP_CONTAINMENT,
    QUERY_GROUP_GLOBAL,
    QUERY_GROUP_SURFACE_BAND,
    _infer_bond_pairs,
    approximate_atomic_union_sdf,
    sample_query_points,
)
from biomol_surface_unsup.geometry.sdf_ops import box_sdf


def test_sample_query_points_returns_hierarchical_groups() -> None:
    torch.manual_seed(0)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
    radii = torch.tensor([1.2, 1.3], dtype=torch.float32)

    sampling = sample_query_points(coords=coords, radii=radii, num_query_points=8, padding=2.0)

    assert tuple(sampling["query_points"].shape) == (8, 3)
    assert tuple(sampling["query_group"].shape) == (8,)
    assert tuple(sampling["containment_points"].shape) == (2, 3)
    assert sampling["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2}
    assert int((sampling["query_group"] == QUERY_GROUP_GLOBAL).sum()) == 4
    assert int((sampling["query_group"] == QUERY_GROUP_CONTAINMENT).sum()) == 2
    assert int((sampling["query_group"] == QUERY_GROUP_SURFACE_BAND).sum()) == 2


def test_sample_query_points_can_add_area_only_uniform_points() -> None:
    torch.manual_seed(0)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
    radii = torch.tensor([1.2, 1.3], dtype=torch.float32)

    sampling = sample_query_points(
        coords=coords,
        radii=radii,
        num_query_points=8,
        num_area_points=5,
        padding=2.0,
    )
    area_points = sampling["query_points"][sampling["query_group"] == QUERY_GROUP_AREA]

    assert tuple(sampling["query_points"].shape) == (13, 3)
    assert sampling["sampling_counts"] == {"global": 4, "containment": 2, "surface_band": 2, "area": 5}
    assert area_points.shape[0] == 5
    assert torch.all(area_points >= sampling["bbox_lower"].unsqueeze(0))
    assert torch.all(area_points <= sampling["bbox_upper"].unsqueeze(0))


def test_surface_band_points_are_close_to_toy_atomic_union_boundary() -> None:
    torch.manual_seed(1)
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
    radii = torch.tensor([1.2, 1.3], dtype=torch.float32)

    sampling = sample_query_points(coords=coords, radii=radii, num_query_points=12, padding=2.0)
    surface_points = sampling["query_points"][sampling["query_group"] == QUERY_GROUP_SURFACE_BAND]
    band_sdf = approximate_atomic_union_sdf(coords, radii, surface_points)

    assert surface_points.shape[0] == sampling["sampling_counts"]["surface_band"]
    assert torch.all(band_sdf.abs() <= 0.5)


def test_loose_box_sampling_uses_box_surface_band_and_domain_volume() -> None:
    torch.manual_seed(3)
    coords = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
    radii = torch.tensor([1.0, 1.0], dtype=torch.float32)

    sampling = sample_query_points(
        coords=coords,
        radii=radii,
        num_query_points=40,
        padding=2.0,
        initialization_mode="loose_box",
        loose_surface_padding=2.0,
        domain_padding=4.0,
        surface_band_width=0.25,
    )
    surface_points = sampling["query_points"][sampling["query_group"] == QUERY_GROUP_SURFACE_BAND]
    global_points = sampling["query_points"][sampling["query_group"] == QUERY_GROUP_GLOBAL]
    surface_phi = box_sdf(surface_points, sampling["surface_bbox_lower"], sampling["surface_bbox_upper"])

    assert torch.all(surface_phi.abs() <= 0.25 + 1e-6)
    assert torch.all(global_points >= sampling["domain_bbox_lower"].unsqueeze(0))
    assert torch.all(global_points <= sampling["domain_bbox_upper"].unsqueeze(0))
    expected_volume = (sampling["domain_bbox_upper"] - sampling["domain_bbox_lower"]).prod()
    assert sampling["bbox_volume"].item() == pytest.approx(expected_volume.item())
    assert torch.allclose(sampling["bbox_lower"], sampling["domain_bbox_lower"])
    assert torch.allclose(sampling["bbox_upper"], sampling["domain_bbox_upper"])


def test_containment_points_cover_interstitial_regions_inside_atomic_union() -> None:
    torch.manual_seed(2)
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 1.2, 0.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor([1.1, 1.1, 1.0], dtype=torch.float32)

    sampling = sample_query_points(coords=coords, radii=radii, num_query_points=12, padding=1.5)
    containment_points = sampling["containment_points"]
    containment_sdf = approximate_atomic_union_sdf(coords, radii, containment_points)

    assert containment_points.shape[0] == sampling["sampling_counts"]["containment"]
    assert torch.all(containment_sdf <= 0.0)


def test_infer_bond_pairs_avoids_dense_all_pairs_materialization_behaviorally() -> None:
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0], [8.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    radii = torch.tensor([0.8, 0.8, 0.8, 0.8], dtype=torch.float32)

    pairs = _infer_bond_pairs(coords, radii, max_neighbors=2, chunk_size=2)

    pair_set = {tuple(pair.tolist()) for pair in pairs}
    assert (0, 1) in pair_set
    assert (1, 2) in pair_set
    assert (0, 3) not in pair_set
    assert all(i < j for i, j in pair_set)
