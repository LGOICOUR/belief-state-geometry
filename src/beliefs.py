"""Analytic belief-state geometry: the *prediction* side of the experiment.

Given a :class:`hmms.base.Process`, this module computes the things the paper
predicts should appear (linearly) in the transformer's residual stream:

* the **Mixed-State Presentation (MSP)** -- the set of reachable optimal-observer
  belief states. Finite processes (RRXOR) are enumerated exactly by BFS over
  emission words; the infinite Mess3 fractal is *sampled* (carry the running
  belief forward along a long trajectory) since it cannot be enumerated.
* the **entropy rate** of the process -- the information-theoretic floor that a
  next-token-prediction loss converges to. Computed exactly from the finite MSP
  when possible, otherwise estimated by Monte-Carlo with the exact optimal
  predictor (which converges to the true value by ergodicity).
* the **in-context optimal loss** -- the achievable loss for a predictor limited
  to a finite context window, which is the apt floor for *this* model's training
  loss (n_ctx = 10).
* a **2-simplex projection** for plotting 3-state beliefs as a triangle, plus an
  RGB coloring (belief == color) used to show theory and recovery match.

The belief update itself lives on the :class:`Process` (``belief_update``,
``belief_trajectory``); this module is the analysis built on top of it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hmms.base import Process

_TOL = 1e-12


# ====================================================================== #
# Information-theory helpers
# ====================================================================== #
def entropy(p: np.ndarray, axis: int = -1, units: str = "bits") -> np.ndarray:
    """Shannon entropy along ``axis``. ``units`` in {'bits','nats'}."""
    p = np.asarray(p, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    h = -terms.sum(axis=axis)
    return h / np.log(2) if units == "bits" else h


# ====================================================================== #
# Mixed-State Presentation (exact enumeration for finite processes)
# ====================================================================== #
@dataclass
class MSP:
    """The enumerated Mixed-State Presentation of a process.

    Attributes
    ----------
    beliefs:        ``[M, n_states]`` the unique belief vectors (states of the MSP).
    emission_probs: ``[M, n_symbols]`` next-symbol distribution at each belief.
    transitions:    ``[M, n_symbols]`` int; next MSP-state index on each symbol
                    (-1 if that symbol has ~0 probability there).
    words:          a representative emission word (list of ints) reaching each belief.
    truncated:      True if enumeration hit ``max_states``/``max_depth`` (i.e. the
                    true MSP is larger or infinite -- as for Mess3).
    process:        the originating process.
    """

    beliefs: np.ndarray
    emission_probs: np.ndarray
    transitions: np.ndarray
    words: list
    truncated: bool
    process: Process

    def __len__(self):
        return self.beliefs.shape[0]

    def stationary(self) -> np.ndarray:
        """Stationary distribution over MSP states (visitation frequency)."""
        M, X = self.emission_probs.shape
        P = np.zeros((M, M))
        for m in range(M):
            for x in range(X):
                nxt = self.transitions[m, x]
                if nxt >= 0:
                    P[m, nxt] += self.emission_probs[m, x]
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        v = np.real(eigvecs[:, idx])
        if v.sum() < 0:
            v = -v
        v = np.clip(v, 0.0, None)
        return v / v.sum()

    def entropy_rate(self, units: str = "bits") -> float:
        """Exact entropy rate = sum_m pi(m) * H(next symbol | belief_m).

        Valid only for a fully-enumerated (non-truncated) MSP.
        """
        if self.truncated:
            raise ValueError(
                "entropy_rate from a truncated MSP is not exact; use entropy_rate_mc "
                "for processes whose MSP is infinite (e.g. Mess3)."
            )
        pi = self.stationary()
        H = entropy(self.emission_probs, axis=1, units=units)
        return float((pi * H).sum())


def enumerate_msp(
    process: Process,
    max_depth: int | None = None,
    max_states: int = 50_000,
    round_dp: int = 9,
) -> MSP:
    """Enumerate reachable belief states by BFS over emission words.

    Starts from the stationary prior (the root of the MSP) and applies the
    belief update for every positive-probability symbol, de-duplicating beliefs
    by rounding to ``round_dp`` decimals.

    For a finite process (RRXOR) this terminates with the complete MSP. For
    Mess3 the MSP is an infinite fractal: pass ``max_depth`` to get the
    depth-limited approximation (the support reachable within that many tokens),
    and check ``.truncated``.
    """
    prior = process.stationary()
    beliefs: list[np.ndarray] = []
    words: list[list[int]] = []
    key_to_idx: dict[tuple, int] = {}

    def add(b, word):
        key = tuple(np.round(b, round_dp))
        if key in key_to_idx:
            return key_to_idx[key], False
        idx = len(beliefs)
        key_to_idx[key] = idx
        beliefs.append(b)
        words.append(word)
        return idx, True

    start_idx, _ = add(prior, [])
    queue = deque([(start_idx, 0)])
    emission: dict[int, np.ndarray] = {}
    transitions: dict[tuple, int] = {}
    truncated = False

    while queue:
        idx, depth = queue.popleft()
        b = beliefs[idx]
        p = process.next_symbol_dist(b)
        emission[idx] = p
        if max_depth is not None and depth >= max_depth:
            continue
        for x in range(process.n_symbols):
            if p[x] <= _TOL:
                continue
            nidx, is_new = add(process.belief_update(b, x), words[idx] + [x])
            transitions[(idx, x)] = nidx
            if is_new:
                if len(beliefs) > max_states:
                    truncated = True
                    break
                queue.append((nidx, depth + 1))
        if truncated:
            break

    M, X = len(beliefs), process.n_symbols
    em = np.zeros((M, X))
    tr = -np.ones((M, X), dtype=int)
    for m in range(M):
        if m in emission:
            em[m] = emission[m]
        for x in range(X):
            if (m, x) in transitions:
                tr[m, x] = transitions[(m, x)]
    return MSP(np.array(beliefs), em, tr, words, truncated, process)


# ====================================================================== #
# Belief clouds by sampling (the only option for the Mess3 fractal)
# ====================================================================== #
def sample_belief_cloud(
    process: Process,
    rng: np.random.Generator,
    n_seqs: int = 2000,
    length: int = 100,
    burn_in: int = 0,
):
    """Visitation-weighted cloud of belief states from sampled trajectories.

    Returns ``(beliefs, emissions)`` where ``beliefs`` is ``[N, n_states]``
    (all positions from ``burn_in`` onward, flattened across sequences) and
    ``emissions`` is the matching ``[N]`` *next emitted symbol* at each belief
    (useful for coloring / sanity checks). This empirical cloud is the stationary
    measure on the MSP attractor -- exactly what the residual-stream activations
    should be compared against for Mess3.
    """
    em = process.sample_batch(n_seqs, length, rng)
    bel = process.belief_trajectory(em)  # [B, L, S], belief after each token
    bel = bel[:, burn_in:, :].reshape(-1, process.n_states)
    nxt = em[:, burn_in:].reshape(-1)
    return bel, nxt


# ====================================================================== #
# Entropy rate / loss floor
# ====================================================================== #
def entropy_rate_mc(
    process: Process,
    rng: np.random.Generator,
    n_seqs: int = 4000,
    length: int = 200,
    burn_in: int = 50,
    units: str = "bits",
) -> float:
    """Monte-Carlo entropy rate via the exact optimal predictor.

    Averages the belief-conditioned next-symbol entropy over sampled
    trajectories (after a burn-in so beliefs reach the stationary measure). This
    converges to the true entropy rate by ergodicity and is the right estimator
    for processes whose MSP is infinite (Mess3).
    """
    bel, _ = sample_belief_cloud(process, rng, n_seqs, length, burn_in)
    nd = np.einsum("ni,xij->nx", bel, process.T)  # P(symbol | belief), [N, X]
    return float(entropy(nd, axis=1, units=units).mean())


def entropy_rate(process: Process, rng: np.random.Generator | None = None,
                 units: str = "bits", max_states: int = 20_000) -> float:
    """Entropy rate: exact from a finite MSP, else Monte-Carlo.

    Tries to enumerate the MSP; if it is finite (not truncated) returns the exact
    value, otherwise falls back to :func:`entropy_rate_mc` (which needs ``rng``).
    """
    msp = enumerate_msp(process, max_states=max_states)
    if not msp.truncated:
        return msp.entropy_rate(units=units)
    if rng is None:
        raise ValueError("process MSP is infinite; pass an rng for the MC estimate.")
    return entropy_rate_mc(process, rng, units=units)


def optimal_in_context_loss(
    process: Process,
    rng: np.random.Generator,
    context_len: int,
    n_seqs: int = 20000,
    units: str = "nats",
) -> float:
    """Optimal achievable next-token loss for a predictor with a finite context.

    This is the apt floor for the transformer's *training* loss: each sequence is
    a fresh window of ``context_len`` tokens starting from the stationary prior,
    and we average the belief-conditioned next-symbol entropy over the positions
    that have a prediction target (predicting tokens 1..L-1 from prefixes
    0..L-2). Defaults to nats to match PyTorch cross-entropy.

    Differs from the asymptotic entropy rate only by the finite-context penalty
    (negligible for Mess3 at n_ctx=10, where the belief nearly synchronises).
    """
    em = process.sample_batch(n_seqs, context_len, rng)
    bel = process.belief_trajectory(em)            # [B, L, S], belief after token t
    nd = np.einsum("bli,xij->blx", bel, process.T)  # [B, L, X]
    # belief after token t predicts token t+1, so use positions t = 0 .. L-2.
    H = entropy(nd[:, :-1, :], axis=2, units=units)  # [B, L-1]
    return float(H.mean())


# ====================================================================== #
# Simplex projection + coloring (for 3-state beliefs -> triangle)
# ====================================================================== #
# Equilateral-triangle vertices for the 2-simplex, ordered (state0, state1, state2).
_TRI_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2.0]])


def project_to_simplex_2d(beliefs: np.ndarray) -> np.ndarray:
    """Map 3-state belief vectors to 2-D barycentric (triangle) coordinates.

    ``beliefs`` is ``[..., 3]``; returns ``[..., 2]``. Used for the Mess3 figure.
    """
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.shape[-1] != 3:
        raise ValueError("project_to_simplex_2d expects 3-state beliefs (last dim 3).")
    return beliefs @ _TRI_VERTS


def simplex_triangle_vertices() -> np.ndarray:
    """The 3 triangle vertices (for drawing the simplex outline)."""
    return _TRI_VERTS.copy()


def belief_to_rgb(beliefs: np.ndarray) -> np.ndarray:
    """Color a belief by itself: 3-state belief -> RGB in [0,1].

    This single convention is used for *both* the predicted cloud and the
    recovered cloud, so a visual match is a like-for-like comparison.
    """
    beliefs = np.asarray(beliefs, dtype=np.float64)
    if beliefs.shape[-1] != 3:
        raise ValueError("belief_to_rgb expects 3-state beliefs (last dim 3).")
    return np.clip(beliefs, 0.0, 1.0)
