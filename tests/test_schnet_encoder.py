from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from biomol_surface_unsup.models.encoders.schnet_encoder import SchNetEncoder


def test_schnet_encoder_shape_and_mask_invariance():
    encoder = SchNetEncoder(in_dim=13, hidden_dim=16, out_dim=12, rbf_dim=4, num_layers=2)
    features = torch.randn(2, 3, 5, 13)
    mask = torch.tensor(
        [
            [[True, True, True, False, False], [True, True, False, False, False], [True, True, True, True, False]],
            [[True, False, False, False, False], [True, True, True, False, False], [True, True, True, True, True]],
        ]
    )
    out = encoder(features, mask)
    assert out.shape == (2, 3, 12)

    perm = torch.tensor([2, 0, 1, 4, 3])
    permuted_features = features[:, :, perm]
    permuted_mask = mask[:, :, perm]
    permuted_out = encoder(permuted_features, permuted_mask)
    assert torch.allclose(out, permuted_out, atol=1e-5, rtol=1e-5)
