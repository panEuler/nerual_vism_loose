from __future__ import annotations

import torch
import torch.nn as nn

from biomol_surface_unsup.features.global_features import GlobalFeatureEncoder
from biomol_surface_unsup.features.local_features import LocalFeatureBuilder
from biomol_surface_unsup.geometry.sdf_ops import box_sdf
from biomol_surface_unsup.models.decoders.film_decoder import FiLMDecoder
from biomol_surface_unsup.models.decoders.sdf_decoder import SDFDecoder
from biomol_surface_unsup.models.decoders.siren_decoder import SirenDecoder
from biomol_surface_unsup.models.encoders.local_deepsets import LocalDeepSetsEncoder
from biomol_surface_unsup.models.encoders.schnet_encoder import SchNetEncoder
from biomol_surface_unsup.models.fusion import concat_fusion
from biomol_surface_unsup.models.positional_encoding import FourierEncoder


def _masked_center(coords: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    squeeze_batch = coords.ndim == 2
    if squeeze_batch:
        coords = coords.unsqueeze(0)
        mask = None if mask is None else mask.unsqueeze(0)
    if mask is None:
        mask = torch.ones(coords.shape[:2], dtype=torch.bool, device=coords.device)
    mask_f = mask.unsqueeze(-1).to(coords.dtype)
    center = (coords * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    if squeeze_batch:
        return center.squeeze(0)
    return center


def _initialize_last_linear_zero(module: nn.Module) -> None:
    """Zero the final Linear layer so a residual decoder starts from no correction."""
    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    if last_linear is None:
        return
    nn.init.zeros_(last_linear.weight)
    if last_linear.bias is not None:
        nn.init.zeros_(last_linear.bias)


class SurfaceModel(nn.Module):
    def __init__(
        self,
        num_atom_types: int,
        cutoff: float = 8.0,
        max_neighbors: int = 64,
        atom_embed_dim: int = 16,
        rbf_dim: int = 16,
        local_hidden_dim: int = 128,
        local_out_dim: int = 128,
        global_hidden_dim: int = 128,
        global_out_dim: int = 128,
        encoder_type: str = "deepsets",
        encoder_num_layers: int = 2,
        decoder_type: str = "mlp",
        decoder_hidden_dim: int = 128,
        decoder_num_layers: int = 3,
        use_fourier_features: bool = True,
        fourier_num_frequencies: int = 6,
        sdf_base_type: str = "none",
        residual_scale: float = 1.0,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        self.sdf_base_type = str(sdf_base_type).lower()
        if self.sdf_base_type not in {"none", "box"}:
            raise ValueError("sdf_base_type must be either 'none' or 'box'")
        self.residual_scale = float(residual_scale)
        self.local_builder = LocalFeatureBuilder(
            num_atom_types=num_atom_types,
            atom_embed_dim=atom_embed_dim,
            rbf_dim=rbf_dim,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
        )
        if encoder_type == "deepsets":
            self.local_encoder = LocalDeepSetsEncoder(
                in_dim=self.local_builder.feature_dim,
                hidden_dim=local_hidden_dim,
                out_dim=local_out_dim,
            )
        elif encoder_type == "schnet":
            self.local_encoder = SchNetEncoder(
                in_dim=self.local_builder.feature_dim,
                hidden_dim=local_hidden_dim,
                out_dim=local_out_dim,
                rbf_dim=rbf_dim,
                num_layers=encoder_num_layers,
            )
        else:
            raise ValueError(f"unsupported encoder_type: {encoder_type}")

        self.global_encoder = GlobalFeatureEncoder(
            num_atom_types=num_atom_types,
            atom_embed_dim=atom_embed_dim,
            hidden_dim=global_hidden_dim,
            out_dim=global_out_dim,
        )
        self.position_encoder = FourierEncoder(d_in=3, n_freq=fourier_num_frequencies) if use_fourier_features else None
        position_dim = 0 if self.position_encoder is None else self.position_encoder.out_dim
        self.decoder_type = decoder_type
        decoder_in_dim = local_out_dim + global_out_dim + position_dim

        if decoder_type == "mlp":
            self.decoder = SDFDecoder(
                in_dim=decoder_in_dim,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
            )
        elif decoder_type == "siren":
            self.decoder = SirenDecoder(
                in_dim=decoder_in_dim,
                hidden_dim=decoder_hidden_dim,
                num_layers=decoder_num_layers,
            )
        elif decoder_type == "film":
            self.decoder = FiLMDecoder(
                local_dim=local_out_dim + position_dim,
                global_dim=global_out_dim,
                hidden_dim=decoder_hidden_dim,
            )
        else:
            raise ValueError(f"unsupported decoder_type: {decoder_type}")
        if zero_init_output:
            _initialize_last_linear_zero(self.decoder)

    @classmethod
    def from_config(cls, model_cfg: dict[str, object], num_atom_types: int) -> "SurfaceModel":
        cfg = dict(model_cfg or {})
        local_cfg = dict(cfg.get("local_builder", {}))
        global_cfg = dict(cfg.get("global_encoder", {}))
        encoder_cfg = dict(cfg.get("local_encoder", {}))
        decoder_cfg = dict(cfg.get("decoder", {}))
        pos_cfg = dict(cfg.get("position_encoding", {}))
        sdf_base_cfg = dict(cfg.get("sdf_base", {}))
        return cls(
            num_atom_types=num_atom_types,
            cutoff=float(local_cfg.get("cutoff", cfg.get("cutoff", 8.0))),
            max_neighbors=int(local_cfg.get("max_neighbors", cfg.get("max_neighbors", 64))),
            atom_embed_dim=int(local_cfg.get("atom_embed_dim", cfg.get("atom_embed_dim", 16))),
            rbf_dim=int(local_cfg.get("rbf_dim", cfg.get("rbf_dim", 16))),
            local_hidden_dim=int(encoder_cfg.get("hidden_dim", cfg.get("hidden_dim", 128))),
            local_out_dim=int(encoder_cfg.get("out_dim", cfg.get("local_out_dim", 128))),
            global_hidden_dim=int(global_cfg.get("hidden_dim", cfg.get("hidden_dim", 128))),
            global_out_dim=int(global_cfg.get("out_dim", cfg.get("global_out_dim", 128))),
            encoder_type=str(encoder_cfg.get("type", cfg.get("encoder_type", "deepsets"))),
            encoder_num_layers=int(encoder_cfg.get("num_layers", 2)),
            decoder_type=str(decoder_cfg.get("type", cfg.get("decoder_type", "mlp"))),
            decoder_hidden_dim=int(decoder_cfg.get("hidden_dim", cfg.get("decoder_hidden_dim", 128))),
            decoder_num_layers=int(decoder_cfg.get("num_layers", 3)),
            use_fourier_features=bool(pos_cfg.get("enabled", cfg.get("use_fourier_features", True))),
            fourier_num_frequencies=int(pos_cfg.get("n_freq", cfg.get("fourier_num_frequencies", 6))),
            sdf_base_type=str(sdf_base_cfg.get("type", cfg.get("sdf_base_type", "none"))),
            residual_scale=float(sdf_base_cfg.get("residual_scale", cfg.get("residual_scale", 1.0))),
            zero_init_output=bool(sdf_base_cfg.get("zero_init_output", cfg.get("zero_init_output", False))),
        )

    def forward(
        self,
        coords,
        atom_types,
        radii,
        query_points,
        charges=None,
        epsilon=None,
        sigma=None,
        atom_mask=None,
        query_mask=None,
        bbox_lower=None,
        bbox_upper=None,
        return_aux=True,
    ):
        local = self.local_builder(
            coords,
            atom_types,
            radii,
            query_points,
            charges=charges,
            epsilon=epsilon,
            sigma=sigma,
            atom_mask=atom_mask,
            query_mask=query_mask,
        )
        z_local = self.local_encoder(local["features"], local["mask"])
        z_global = self.global_encoder(
            coords,
            atom_types,
            radii,
            charges=charges,
            epsilon=epsilon,
            sigma=sigma,
            atom_mask=atom_mask,
        )
        if z_local.ndim == 2:
            z_global_expanded = z_global.unsqueeze(0).expand(query_points.shape[0], -1)
        else:
            z_global_expanded = z_global.unsqueeze(1).expand(-1, z_local.shape[1], -1)

        position_features = None
        if self.position_encoder is not None:
            center = _masked_center(coords, atom_mask)
            rel_query = query_points - center
            position_features = self.position_encoder(rel_query)

        if self.decoder_type == "film":
            local_input = z_local if position_features is None else concat_fusion(z_local, position_features)
            raw_residual = self.decoder(local_input, z_global_expanded)
            fused = concat_fusion(local_input, z_global_expanded)
        else:
            fused = concat_fusion(z_local, z_global_expanded)
            if position_features is not None:
                fused = concat_fusion(fused, position_features)
            raw_residual = self.decoder(fused)

        if self.sdf_base_type == "box":
            if bbox_lower is None or bbox_upper is None:
                raise ValueError("bbox_lower and bbox_upper are required when sdf_base.type='box'")
            base_phi = box_sdf(query_points, bbox_lower, bbox_upper)
            sdf = base_phi + self.residual_scale * raw_residual
        else:
            base_phi = None
            sdf = raw_residual

        if query_mask is not None:
            sdf = sdf * query_mask.to(sdf.dtype)
        output = {"sdf": sdf}
        if self.sdf_base_type == "box":
            output["raw_residual"] = raw_residual
            output["base_phi"] = base_phi
        if return_aux:
            output.update({"z_local": z_local, "z_global": z_global_expanded, "fused": fused, **local})
        return output
