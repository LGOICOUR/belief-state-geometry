"""Model construction: a small ``HookedTransformer`` matching the paper.

We use TransformerLens's :class:`HookedTransformer` specifically because the
residual stream is first-class: ``blocks.{l}.hook_resid_post`` (and ``_pre``,
``_mid``) are cache keys we can read directly, which is the whole point of the
probe. The architecture is the paper's (arXiv:2405.15943, App. A.4):

    n_layers=4, d_model=64, n_heads=1, d_head=8, d_mlp=256, n_ctx=10,
    LayerNorm, ReLU, causal attention.

Positional encoding is unspecified in the paper; we use TransformerLens's
standard learned absolute positional embeddings (noted in the README).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from hmms.base import Process


@dataclass
class ArchConfig:
    """Architecture hyperparameters (paper defaults; ``d_vocab`` set per process)."""

    d_vocab: int
    n_layers: int = 4
    d_model: int = 64
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256
    n_ctx: int = 10
    act_fn: str = "relu"
    normalization_type: str = "LN"          # LayerNorm
    attention_dir: str = "causal"
    positional_embedding_type: str = "standard"  # learned absolute (paper unspecified)
    seed: int = 0

    @classmethod
    def paper(cls, process: Process, **overrides) -> "ArchConfig":
        """Paper architecture with vocabulary size taken from the process."""
        cfg = cls(d_vocab=process.n_symbols)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def to_tl_config(self) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_ctx=self.n_ctx,
            d_head=self.d_head,
            n_heads=self.n_heads,
            d_mlp=self.d_mlp,
            d_vocab=self.d_vocab,
            act_fn=self.act_fn,
            normalization_type=self.normalization_type,
            attention_dir=self.attention_dir,
            positional_embedding_type=self.positional_embedding_type,
            seed=self.seed,
        )

    def as_dict(self) -> dict:
        return asdict(self)


def build_model(arch: ArchConfig, device: str | torch.device = "cpu") -> HookedTransformer:
    """Instantiate a randomly-initialised HookedTransformer on ``device``."""
    model = HookedTransformer(arch.to_tl_config())
    return model.to(device)


def n_params(model: HookedTransformer) -> int:
    return sum(p.numel() for p in model.parameters())
