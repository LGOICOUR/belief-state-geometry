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
    init_states=None,
) -> torch.Tensor:
    """Sample ``[n_seqs, seq_len]`` integer token tensor from the process.

    ``init_states`` (optional, shape ``[n_seqs]``) seeds the hidden state of each
    sequence; defaults to draws from the stationary distribution. The Phase-2
    mixture uses it for epoch-aligned sampling.
    """
    em = process.sample_batch(n_seqs, seq_len, rng, init_states=init_states)
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


# --------------------------------------------------------------------- #
# Phase 2: epoch-aligned eval set for the mixture process
# --------------------------------------------------------------------- #
@dataclass
class MixtureEvalData:
    """Held-out set for the mixture experiments.

    Attributes
    ----------
    tokens:       ``[n_seqs, seq_len]`` LongTensor (on device).
    z_labels:     ``[n_seqs, seq_len]`` int; the generator (0=A, 1=B) of the epoch
                  each position belongs to (ground truth from the hidden-state path).
    gen_marginal: ``[n_seqs, seq_len]`` float; the optimal observer's
                  ``P(Z=A | history)`` after each token (epoch-aligned prior).
    emissions:    ``[n_seqs, seq_len]`` numpy copy of the tokens.
    """

    tokens: torch.Tensor
    z_labels: np.ndarray
    gen_marginal: np.ndarray
    emissions: np.ndarray

    @property
    def z_first(self) -> np.ndarray:
        """The first epoch's generator label, one per sequence (for retention probes)."""
        return self.z_labels[:, 0]


def make_mixture_eval_set(process, n_seqs, seq_len, seed, device="cpu") -> MixtureEvalData:
    """Sample an *epoch-aligned* held-out set for the mixture process.

    Every sequence starts at an epoch boundary (so within-epoch phase equals
    ``position % epoch_len``). Attaches both the ground-truth generator label per
    position (from the hidden-state path) and the optimal observer's generator
    marginal (from the epoch-aligned belief trajectory). Requires a process with
    ``aligned_init_states``, ``generator_of_state``, ``generator_marginal`` and
    ``epoch_start_belief`` (i.e. a :class:`hmms.MixtureProcess`).
    """
    rng = np.random.default_rng(seed)
    init = process.aligned_init_states(n_seqs, rng)
    em, states = process.sample_batch(
        n_seqs, seq_len, rng, init_states=init, return_states=True
    )
    z_labels = process.generator_of_state[states[:, :seq_len]]  # gen of the emitter of each token
    beliefs = process.belief_trajectory(em, start=process.epoch_start_belief())
    gen_marginal = process.generator_marginal(beliefs)          # [n_seqs, seq_len]
    tokens = torch.as_tensor(em, dtype=torch.long, device=device)
    return MixtureEvalData(tokens=tokens, z_labels=z_labels,
                           gen_marginal=gen_marginal, emissions=em)
