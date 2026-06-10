"""Phase 2: the mixture process for the belief-state collapse experiment.

A **mixture process** interleaves two generators, ``A`` and ``B``, that are
*predictively identical* for a tunable horizon and then diverge. It is a single
:class:`Process` (a labeled transition tensor), so all the Phase-1 belief
machinery -- ``stationary``, ``belief_update``, ``belief_trajectory``, MSP
enumeration, entropy rate -- works on it unchanged. The only new piece is the
**generator marginal** ``P(Z = A | history)``, the "which generator" probe target.

Construction
------------
The stream is a sequence of fixed-length **epochs** of length ``horizon + tail``.
Each epoch draws a fresh fair generator label ``Z in {A, B}`` and emits:

* positions ``1 .. horizon``            -- a uniform random bit (identical under A and B);
* positions ``horizon+1 .. horizon+tail`` -- ``Z``'s **indicator** bit
  (``A -> 0``, ``B -> 1``), emitted deterministically.

At the epoch boundary ``Z`` is resampled. Hence A and B are statistically
indistinguishable for the first ``horizon`` steps (both uniform), and predictive
equivalence breaks at step ``horizon + 1`` (the first indicator).

Why a ``tail >= 2`` (and not a single reveal)
---------------------------------------------
If the divergence were a *single* token at ``horizon+1`` that also ended the
epoch, the generator label would become predictively irrelevant the instant it
is revealed, so the optimal belief would never actually *commit* to it -- there
would be no regime in which representing ``Z`` is predictively *necessary*, only
the question of whether the model *retains* an already-useless fact. A short
``Z``-dependent tail fixes this: during the tail (positions ``horizon+1`` ..
``horizon+tail``) ``Z`` is both **known** (revealed by the first indicator) and
**still predictive** (it determines the remaining indicators), so the optimal
belief legitimately commits. The subsequent epoch reset then makes ``Z``
irrelevant, which is where the *retention* question (experiment E3) lives. So
``tail >= 2`` cleanly separates "represents the predictive belief" (E1/E2) from
"retains predictively-useless identity" (E3).

What the optimal observer does (epoch-aligned)
----------------------------------------------
``P(Z = A | history)`` is ``1/2`` through the random prefix (no evidence),
jumps to ``0`` or ``1`` at the first indicator (the **bifurcation**), holds
through the tail, and resets to ``1/2`` at the next epoch (``Z`` discarded).
"""

from __future__ import annotations

import numpy as np

from .base import Process

# Generator labels and the bit each emits in its tail.
GEN_A, GEN_B = 0, 1
_INDICATOR = {GEN_A: 0, GEN_B: 1}  # A's tail is all 0s, B's tail is all 1s


def mixture_tensor(horizon: int, tail: int = 2):
    """Build ``(T, state_names, generator_of_state)`` for the mixture process.

    Parameters
    ----------
    horizon: number of shared uniform-random bits per epoch (>= 1). This is the
        divergence horizon ``i``: A and B are identical for these steps.
    tail:    number of ``Z``-indicator bits per epoch (>= 1; use >= 2 so the
        belief commits -- see the module docstring).

    Returns
    -------
    T : ``[2, n_states, n_states]`` labeled transition tensor.
    state_names : human-readable labels, e.g. ``"A.pre1"``, ``"B.tail2"``.
    generator_of_state : ``[n_states]`` int array, ``GEN_A``/``GEN_B`` per state.

    State layout (generator-major, so the first half are A-states):
        A.pre1 .. A.pre{horizon}, A.tail1 .. A.tail{tail},
        B.pre1 .. B.pre{horizon}, B.tail1 .. B.tail{tail}.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if tail < 1:
        raise ValueError("tail must be >= 1")

    per_gen = horizon + tail            # states per generator
    n_states = 2 * per_gen
    n_symbols = 2

    def pre(Z, pos):                    # pos in 1..horizon
        return Z * per_gen + (pos - 1)

    def tl(Z, k):                       # k in 1..tail
        return Z * per_gen + horizon + (k - 1)

    T = np.zeros((n_symbols, n_states, n_states), dtype=np.float64)
    for Z in (GEN_A, GEN_B):
        # Prefix: emit a uniform random bit, advance the phase. Both bits lead to
        # the same next state, so the prefix carries no information about Z.
        for pos in range(1, horizon + 1):
            nxt = pre(Z, pos + 1) if pos < horizon else tl(Z, 1)
            for x in (0, 1):
                T[x, pre(Z, pos), nxt] += 0.5
        # Tail: emit Z's indicator deterministically.
        ind = _INDICATOR[Z]
        for k in range(1, tail + 1):
            if k < tail:
                T[ind, tl(Z, k), tl(Z, k + 1)] += 1.0
            else:  # epoch boundary: resample Z fairly
                T[ind, tl(Z, k), pre(GEN_A, 1)] += 0.5
                T[ind, tl(Z, k), pre(GEN_B, 1)] += 0.5

    state_names: list[str] = []
    generator_of_state = np.empty(n_states, dtype=int)
    for Z in (GEN_A, GEN_B):
        tag = "A" if Z == GEN_A else "B"
        for pos in range(1, horizon + 1):
            generator_of_state[pre(Z, pos)] = Z
            state_names.append(f"{tag}.pre{pos}")
        for k in range(1, tail + 1):
            generator_of_state[tl(Z, k)] = Z
            state_names.append(f"{tag}.tail{k}")
    return T, state_names, generator_of_state


class MixtureProcess(Process):
    """Two generators, predictively identical for ``horizon`` steps then divergent.

    The generator label ``Z`` is resampled fairly each epoch. Use
    :meth:`generator_marginal` to get the "which generator" probe target
    ``P(Z = A | belief)`` from any belief (single or batched).
    """

    def __init__(self, horizon: int = 2, tail: int = 2):
        T, state_names, gen_of_state = mixture_tensor(horizon, tail)
        super().__init__(
            T,
            symbol_names=["0", "1"],
            state_names=state_names,
            name=f"Mixture(h={horizon},tail={tail})",
        )
        self.horizon = horizon
        self.tail = tail
        self.epoch_len = horizon + tail
        self.generator_of_state = gen_of_state          # 0 (A) / 1 (B) per state
        self._gen_a_mask = gen_of_state == GEN_A
        # The two epoch-start states (pre1 of each generator), for aligned seeding.
        self.epoch_start_states = (0, self.epoch_len)    # pre(A,1), pre(B,1)

    def generator_marginal(self, belief: np.ndarray) -> np.ndarray:
        """``P(generator = A | belief)``. Accepts belief shape ``[..., n_states]``;
        returns shape ``[...]`` (a scalar for a single belief)."""
        belief = np.asarray(belief, dtype=np.float64)
        return belief[..., self._gen_a_mask].sum(axis=-1)

    def epoch_start_belief(self) -> np.ndarray:
        """Optimal belief at an epoch boundary: 1/2 on each generator's first state.

        This is the correct prior for *epoch-aligned* analysis (the observer knows
        the phase from absolute position), as opposed to the stationary prior which
        also averages over phase uncertainty.
        """
        b = np.zeros(self.n_states)
        b[self.epoch_start_states[GEN_A]] = 0.5
        b[self.epoch_start_states[GEN_B]] = 0.5
        return b

    def aligned_init_states(self, n_seqs: int, rng: np.random.Generator) -> np.ndarray:
        """Initial hidden states for epoch-aligned sampling: a fair coin over the
        two generators' epoch-start states, so every sequence begins a fresh epoch
        and within-epoch phase equals ``position % epoch_len``."""
        return rng.choice(self.epoch_start_states, size=n_seqs)
