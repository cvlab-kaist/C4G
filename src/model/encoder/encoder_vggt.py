from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.types import BatchedExample, DataShim
from .heads import DPTHead
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .common.gmae import Transformer, InstillTransformer
from .backbone.croco.misc import fill_default_args, freeze_all_params
from ..encodings.positional_encoding import SinusoidalPositionalEmbedding


inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


def zero_init_(layer: nn.Linear):
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)

@dataclass
class EncoderVGGTCfg:
    name: Literal["vggt"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_gaussians: int = 2048
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    freeze_backbone: bool = False
    decoder_depth: int = 2
    gaussians_per_token: int = 1
    timestamp_embedding: bool = True
    sinusoidal_embedding_dim: Optional[int] = 256
    gradient_checkpoint: bool = True
    sinusoidal_period: Optional[int] = 10000


class EncoderVGGT(Encoder[EncoderVGGTCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderVGGTCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3, cfg.gradient_checkpoint)

        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.aggregator.patch_size
        self.raw_gs_dim = 3 + 1 + self.gaussian_adapter.d_in

        self.dpt_head = DPTHead(2048) if 'vggt' in cfg.backbone.name else None
        freeze_all_params([self.dpt_head])
        if hasattr(self.backbone, 'camera_head'):
            del self.backbone.camera_head, self.backbone.point_head, self.backbone.depth_head, self.backbone.track_head

        if cfg.freeze_backbone:
            self.backbone.set_freeze('encoder')
                    
        transformer_dim = 2048
        
        self.gaussian_tokens = nn.Parameter(torch.randn(cfg.num_gaussians, transformer_dim))
        self.anchor_positions = nn.Parameter(torch.tensor([[0,0,1]]).repeat(cfg.num_gaussians,1), requires_grad=False)
        
        if cfg.timestamp_embedding:
            self.time_embedding = SinusoidalPositionalEmbedding(cfg.sinusoidal_embedding_dim, max_period=cfg.sinusoidal_period)
            self.time_mlp = nn.ModuleList([
                nn.Linear(cfg.sinusoidal_embedding_dim, cfg.sinusoidal_embedding_dim),
                nn.ReLU(),
                nn.Linear(cfg.sinusoidal_embedding_dim, transformer_dim),
                nn.ReLU()
            ])

        self.gmae_decoder = Transformer(
            dim = transformer_dim,
            depth = cfg.decoder_depth,
            heads = 16,
            dim_head = transformer_dim//16,
            mlp_dim = transformer_dim * 2,
            cfg = cfg,
            rope = None
        )
        self.gmae_to_gaussians = nn.Linear(transformer_dim, self.raw_gs_dim * cfg.gaussians_per_token)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        context_feature: Optional[Tensor] = None,
        target_timestamps: Optional[Tensor] = None,
    ) -> dict[int, Gaussians]:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.freeze_backbone:
            with torch.no_grad():
                dec, shape, patch_start_idx = self.backbone(context, return_views=False)
        else:
            dec, shape, patch_start_idx = self.backbone(context, return_views=False)
            
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                res = self.dpt_head(dec, context['image'], patch_start_idx)
                vis_depth = res[0][..., -1]
            
        dec_feat = dec[-1][:, :, patch_start_idx:]
        b, v, n, d = dec_feat.shape
                
        dec_feat = rearrange(dec_feat, "b v n d -> b (v n) d")
        
        # Prepare gaussian tokens with timestamp embedding
        gaussian_tokens_batch = self.gaussian_tokens.unsqueeze(0).expand(b, -1, -1)
        all_decoder_tokens = torch.cat((dec_feat, gaussian_tokens_batch), dim=1)
        gaussians_per_timestamp = {}

        if target_timestamps is not None:
            save_attn_for_first = visualization_dump is not None
            if save_attn_for_first and not self.training:
                import random
                attn_save_idx = random.randint(0, len(target_timestamps) - 1)
            else:
                attn_save_idx = -1
            for i, target_timestamp in enumerate(target_timestamps):
                t_idx = target_timestamp.item()
                do_save_attn = save_attn_for_first and (i == attn_save_idx) and (not self.training)

                if self.training:
                    means, covs, harmonics, opacs, scales, rots = ckpt_fn(
                        self._decode_one_timestamp,
                        all_decoder_tokens, context['index'], target_timestamp,
                        b, n, global_step, device,
                        use_reentrant=False,
                    )
                else:
                    result = self._decode_one_timestamp(
                        all_decoder_tokens, context['index'], target_timestamp,
                        b, n, global_step, device,
                        save_attention=do_save_attn,
                    )
                    if do_save_attn:
                        (means, covs, harmonics, opacs, scales, rots), attn_weights = result
                        visualization_dump['attention'] = attn_weights  # list of (B, heads, seq, seq)
                        visualization_dump['attention_n_patches_per_view'] = n
                        visualization_dump['attention_num_views'] = v
                        visualization_dump['attention_context_index'] = context['index'].clone()
                        visualization_dump['attention_target_timestamp'] = target_timestamp.item()
                    else:
                        means, covs, harmonics, opacs, scales, rots = result

                gaussians_per_timestamp[t_idx] = Gaussians(means, covs, harmonics, opacs, scales, rots)

            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump['depth'] = vis_depth.unsqueeze(-1).unsqueeze(-1)
                visualization_dump["scales"] = None
                visualization_dump["rotations"] = None
                visualization_dump["means"] = None
                visualization_dump['opacities'] = None

        return gaussians_per_timestamp

    def _decode_one_timestamp(self, all_decoder_tokens, context_index, target_timestamp, b, n, global_step, device, save_attention=False):
        """Decode gaussians for a single timestamp. Wrapped by gradient checkpoint during training."""
        if save_attention:
            self.gmae_decoder.enable_attention_save()
        if self.cfg.timestamp_embedding:
            ctx_ts = repeat(context_index, 'b v -> (b v n)', n=n)
            tgt_ts = repeat(target_timestamp, ' -> (b n)', b=b, n=self.cfg.num_gaussians)
            timesteps = torch.cat((ctx_ts, tgt_ts), dim=0)
            rel_timesteps = (timesteps - ctx_ts.min()) / (tgt_ts.max() - ctx_ts.min() + 1e-8)
            timestep_emb = self.time_embedding(rel_timesteps).to(device)
            for m in self.time_mlp:
                timestep_emb = m(timestep_emb)
            timestep_emb = rearrange(timestep_emb, "(b n) d -> b n d", b=b)
            tokens_with_emb = all_decoder_tokens + timestep_emb
            decoded_tokens = self.gmae_decoder(tokens_with_emb)

        else:
            decoded_tokens = self.gmae_decoder(all_decoder_tokens)

        attn_weights = None
        if save_attention:
            attn_weights = self.gmae_decoder.get_attention_weights()
            self.gmae_decoder.disable_attention_save()

        gaussian_params = self.gmae_to_gaussians(decoded_tokens[:, -self.gaussian_tokens.shape[0]:])
        gaussian_params = rearrange(gaussian_params, "b n (gpt c) -> b (n gpt) c", gpt=self.cfg.gaussians_per_token, c=self.raw_gs_dim)

        pts_all = gaussian_params[:, :, :3].unsqueeze(-2) + self.anchor_positions.unsqueeze(dim=0).repeat(b, self.cfg.gaussians_per_token, 1).unsqueeze(dim=2)
        depths = pts_all[..., -1].unsqueeze(-1)

        gaussians = gaussian_params[:, :, 3:]
        gaussians = rearrange(gaussians, "... c -> ... () c")
        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            self.map_pdf_to_opacity(densities, global_step),
            rearrange(gaussians[..., 1:], "b n d c -> b n d () c"),
        )

        result = (
            rearrange(gaussians.means, "b n srf spp xyz -> b (n srf spp) xyz"),
            rearrange(gaussians.covariances, "b n srf spp i j -> b (n srf spp) i j"),
            rearrange(gaussians.harmonics, "b n srf spp c d_sh -> b (n srf spp) c d_sh"),
            rearrange(gaussians.opacities, "b n srf spp -> b (n srf spp)"),
            rearrange(gaussians.scales, "b n srf spp d -> b (n srf spp) d"),
            rearrange(gaussians.rotations, "b n srf spp q -> b (n srf spp) q"),
        )

        if save_attention:
            return result, attn_weights
        return result

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
