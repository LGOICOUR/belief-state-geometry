"""Data pipeline: turn a :class:`Process` into token batches for the transformer.

The paper trains *online*: every step it samples a fresh batch of 64 sequences of
length ``n_ctx`` from the HMM, each seeded from the stationary state. We mirror
that here -- there is no fixed training set on disk. With only ``n_symbols ** n_ctx``
distinct windows (~59k for Mess3 at n_ctx=10) the model effectively learns the
*whole* distribution, so the relevant "held-out" discipline is for the **probe**:
fit the linear map on one freshly-sampled activation set and report R^2 on a
*disjoint* one (different seed). :func:`make_eval_set` builds such a set together
with the ground-truth optimal beliefs the probe regresses onto.

Everything is generic over the process, so the Phase-2 mixture pipeline reuses it
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hmms.base import Process


# --------------------------------------------------------------------- #
# Runtime helpers (device + reproducible seeding)
# --------------------------------------------------------------------- #
def get_device(prefer: str | None = None) -> torch.device:
    """Pick a device: explicit ``prefer``, else MPS (Apple Silicon) > CUDA > CPU."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python/NumPy/PyTorch RNGs and return a NumPy Generator for sampling.

    The returned Generator is what we thread explicitly through all HMM sampling;
    the global torch/numpy seeds make model init and any incidental randomness
    reproducible too.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


# --------------------------------------------------------------------- #
# Token sampling
# --------------------------------------------------------------------- #
def sample_tokens(
    process: Process,
    n_seqs: int,
    seq_len: int,
    rng: np.random.Generator,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample ``[n_seqs, seq_len]`` integer token tensor from the process."""
    em = process.sample_batch(n_seqs, seq_len, rng)
    return torch.as_tensor(em, dtype=torch.long, device=device)


@dataclass
class EvalData:
    """A held-out evaluation set for the probe.

    Attributes
    ----------
    tokens:    ``[n_seqs, seq_len]`` LongTensor (on the model's device).
    beliefs:   ``[n_seqs, seq_len, n_states]`` ground-truth optimal beliefs
               (numpy), aligned so ``beliefs[b, t]`` is the belief *after* token t
               -- i.e. the sufficient statistic in the residual stream at position t.
    emissions: ``[n_seqs, seq_len]`` numpy copy of the tokens (for convenience).
    """

    tokens: torch.Tensor
    beliefs: np.ndarray
    emissions: np.ndarray


def make_eval_set(
    process: Process,
    n_seqs: int,
    seq_len: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> EvalData:
    """Sample a held-out set and attach analytic optimal beliefs at every position."""
    rng = np.random.default_rng(seed)
    em = process.sample_batch(n_seqs, seq_len, rng)
    beliefs = process.belief_trajectory(em)  # [n_seqs, seq_len, n_states]
    tokens = torch.as_tensor(em, dtype=torch.long, device=device)
    return EvalData(tokens=tokens, beliefs=beliefs, emissions=em)
