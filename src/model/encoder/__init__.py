from typing import Optional

from .encoder import Encoder
from .encoder_vggt import EncoderVGGT, EncoderVGGTCfg
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "vggt": (EncoderVGGT, None),
}

EncoderCfg = EncoderVGGTCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
