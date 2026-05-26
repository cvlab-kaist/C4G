import math

import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torch as th
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """For the sake of simplicity, this encodes values in the range [0, 1]."""

    frequencies: Float[Tensor, "frequency phase"]
    phases: Float[Tensor, "frequency phase"]

    def __init__(self, num_octaves: int):
        super().__init__()
        octaves = torch.arange(num_octaves).float()

        # The lowest frequency has a period of 1.
        frequencies = 2 * torch.pi * 2**octaves
        frequencies = repeat(frequencies, "f -> f p", p=2)
        self.register_buffer("frequencies", frequencies, persistent=False)

        # Choose the phases to match sine and cosine.
        phases = torch.tensor([0, 0.5 * torch.pi], dtype=torch.float32)
        phases = repeat(phases, "p -> f p", f=num_octaves)
        self.register_buffer("phases", phases, persistent=False)

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        samples = einsum(samples, self.frequencies, "... d, f p -> ... d f p")
        return rearrange(torch.sin(samples + self.phases), "... d f p -> ... (d f p)")

    def d_out(self, dimensionality: int):
        return self.frequencies.numel() * dimensionality

## from https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L110
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, device=torch.device("cpu"), max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
        half = embedding_dim // 2
        self.freqs = nn.Parameter(torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ),requires_grad=False)

    def forward(self, timesteps: Float[Tensor, "batch"]):
        try:
            args = timesteps[:, None].float() * self.freqs[None]
        except:
            args = timesteps[:, None].float() * self.freqs[None].to(device=timesteps.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.embedding_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding