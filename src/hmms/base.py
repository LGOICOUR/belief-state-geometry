"""Base class for the data-generating processes (Hidden Markov Models written in
computational-mechanics *labeled-transition* form).

Everything the project needs from a process -- the optimal-observer belief
update, the stationary distribution, sequence sampling -- is derived generically
from a single object: the labeled transition tensor ``T``, where

    T[x, i, j] = P(next hidden state = j  AND  emit symbol x | current state = i).

Summing over the emission symbol recovers the ordinary (row-stochastic)
state-transition matrix::

    sum_x T[x] = T_full,   with every row of T_full summing to 1.

Concrete processes (Mess3, RRXOR, and the Phase-2 Mixture) only need to *build*
this tensor and hand it to ``Process``; all of the belief-state machinery is
then inherited unchanged. This is deliberate: the planned Phase-2
"which-generator" experiments require nothing more than a process whose ``T``
has the right block structure, so keeping every algorithm a pure function of
``T`` is what lets a mixture process slot in without a rewrite.

Conventions
-----------
* Beliefs are row vectors over hidden states (shape ``[n_states]``), living in
  the probability simplex (non-negative, sum to 1).
* Symbols are integer ids in ``range(n_symbols)``; this id *is* the token id the
  transformer sees. The model never observes hidden states.
* All randomness flows through an explicit ``numpy.random.Generator`` for
  reproducibility -- no hidden global RNG state.
"""

from __future__ import annotations

import numpy as np

# Numerical tolerance used when checking that distributions sum to 1, when
# de-duplicating belief vectors, and when guarding against division by ~0.
_TOL = 1e-9


class Process:
    """A discrete Hidden Markov Model in labeled-transition form.

    Parameters
    ----------
    T:
        Array of shape ``[n_symbols, n_states, n_states]`` with
        ``T[x, i, j] = P(next state j, emit x | state i)``. Its sum over the
        symbol axis must be a row-stochastic matrix.
    symbol_names:
        Optional human-readable labels for emission symbols (e.g. ``["A","B","C"]``).
        Purely cosmetic -- used in plots and ``__repr__``.
    state_names:
        Optional labels for hidden states.
    name:
        Optional name for the process (used in plots / logging).
    validate:
        If True (default), check shape and row-stochasticity at construction.
    """

    def __init__(self, T, symbol_names=None, state_names=None, name=None, validate=True):
        T = np.asarray(T, dtype=np.float64)
        if T.ndim != 3 or T.shape[1] != T.shape[2]:
            raise ValueError(
                f"T must have shape [n_symbols, n_states, n_states]; got {T.shape}"
            )
        self.T = T
        self.n_symbols, self.n_states, _ = T.shape
        self.name = name or self.__class__.__name__
        self.symbol_names = (
            list(symbol_names) if symbol_names is not None
            else [str(i) for i in range(self.n_symbols)]
        )
        self.state_names = (
            list(state_names) if state_names is not None
            else [str(i) for i in range(self.n_states)]
        )
        if len(self.symbol_names) != self.n_symbols:
            raise ValueError("symbol_names length must equal n_symbols")
        if len(self.state_names) != self.n_states:
            raise ValueError("state_names length must equal n_states")

        self._stationary = None  # cached lazily
        if validate:
            self._validate()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _validate(self):
        if np.any(self.T < -_TOL):
            raise ValueError("T has negative entries; not a valid probability tensor.")
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError(
                "sum_x T[x] must be row-stochastic (rows sum to 1); "
                f"got row sums {row_sums}."
            )

    # ------------------------------------------------------------------ #
    # Core matrices
    # ------------------------------------------------------------------ #
    @property
    def transition_matrix(self) -> np.ndarray:
        """The full (symbol-marginalised) state-transition matrix ``sum_x T[x]``."""
        return self.T.sum(axis=0)

    def stationary(self) -> np.ndarray:
        """Stationary distribution of the hidden-state chain.

        This is the optimal observer's prior belief *before seeing any symbol*.
        Computed as the left eigenvector of the transition matrix for
        eigenvalue 1, normalised to a probability vector.
        """
        if self._stationary is None:
            Tf = self.transition_matrix
            eigvals, eigvecs = np.linalg.eig(Tf.T)  # left eigvecs of Tf == right of Tf.T
            idx = int(np.argmin(np.abs(eigvals - 1.0)))
            v = np.real(eigvecs[:, idx])
            if v.sum() < 0:
                v = -v
            v = np.clip(v, 0.0, None)
            self._stationary = v / v.sum()
        return self._stationary.copy()

    # Alias that reads well at call sites in the belief code.
    def prior_belief(self) -> np.ndarray:
        """The optimal observer's prior belief (== the stationary distribution)."""
        return self.stationary()

    # ------------------------------------------------------------------ #
    # Optimal Bayesian belief update
    # ------------------------------------------------------------------ #
    def belief_update(self, belief: np.ndarray, symbol: int) -> np.ndarray:
        """One step of exact Bayesian filtering over hidden states.

        Given the current belief ``b`` (distribution over hidden states) and an
        observed ``symbol`` x, returns the posterior belief::

            b'      = b @ T[x]          # joint mass on (next state, symbol x)
            b_new   = b' / b'.sum()     # renormalise by P(symbol | b)

        Raises ``ValueError`` if the symbol has probability ~0 under ``belief``
        (which never happens for a symbol that was actually emitted, but can
        happen during exhaustive MSP enumeration -- callers there guard for it).
        """
        b = np.asarray(belief, dtype=np.float64)
        unnorm = b @ self.T[symbol]
        total = unnorm.sum()
        if total <= _TOL:
            raise ValueError(
                f"symbol {symbol} has ~0 probability under the given belief; "
                "cannot normalise."
            )
        return unnorm / total

    def next_symbol_dist(self, belief: np.ndarray) -> np.ndarray:
        """P(next symbol | belief), shape ``[n_symbols]``.

        This is the *optimal next-token distribution*. Note it is strictly
        downstream of the belief: the belief is the sufficient statistic, the
        token distribution is a (generally many-to-one) function of it.
        ``P(x | b) = sum_{i,j} b_i T[x, i, j]``.
        """
        b = np.asarray(belief, dtype=np.float64)
        return np.einsum("i,xij->x", b, self.T)

    def word_to_belief(self, word, start=None) -> np.ndarray:
        """Belief after observing a sequence of symbols (a "word").

        Starts from ``start`` (default: the stationary prior) and applies one
        ``belief_update`` per symbol. With ``start=None`` this returns exactly
        the optimal observer's belief given the whole word as history.
        """
        b = self.stationary() if start is None else np.asarray(start, dtype=np.float64)
        for x in word:
            b = self.belief_update(b, int(x))
        return b

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample_batch(
        self,
        n_seqs: int,
        length: int,
        rng: np.random.Generator,
        init_states=None,
        return_states: bool = False,
    ):
        """Sample a batch of emission sequences, vectorised across sequences.

        Parameters
        ----------
        n_seqs, length:
            Number of sequences and tokens per sequence.
        rng:
            A ``numpy.random.Generator``.
        init_states:
            Optional array of initial hidden-state ids (shape ``[n_seqs]``).
            Defaults to i.i.d. draws from the stationary distribution -- matching
            the paper, which seeds each sequence from the stationary state.
        return_states:
            If True, also return the hidden-state path (shape ``[n_seqs, length+1]``,
            including the initial state). Hidden states are *only* for analysis/
            debugging; the model never sees them.

        Returns
        -------
        emissions ``[n_seqs, length]`` (int64), or ``(emissions, states)``.

        Notes
        -----
        The inner loop runs over ``length`` (10 in the paper) and is fully
        vectorised across the ``n_seqs`` sequences, so sampling a large batch of
        short sequences -- exactly the training regime -- is fast. We sample the
        joint ``(symbol, next_state)`` per step via inverse-CDF, then split the
        flat category into ``symbol = idx // n_states`` and
        ``next_state = idx % n_states``.
        """
        S, X = self.n_states, self.n_symbols
        if init_states is None:
            states = rng.choice(S, size=n_seqs, p=self.stationary())
        else:
            states = np.asarray(init_states, dtype=np.int64).copy()
            if states.shape != (n_seqs,):
                raise ValueError(f"init_states must have shape ({n_seqs},)")

        emissions = np.empty((n_seqs, length), dtype=np.int64)
        states_out = None
        if return_states:
            states_out = np.empty((n_seqs, length + 1), dtype=np.int64)
            states_out[:, 0] = states

        # joint[i] = flattened P(symbol, next_state | state i), order = symbol-major.
        joint = np.transpose(self.T, (1, 0, 2)).reshape(S, X * S)  # [S, X*S]
        cdf = np.cumsum(joint, axis=1)
        cdf[:, -1] = 1.0  # guard against fp drift so r < cdf always matches

        for t in range(length):
            r = rng.random(n_seqs)
            # first index where r < cdf[state]  ==  inverse-CDF sample
            idx = (r[:, None] < cdf[states]).argmax(axis=1)
            emissions[:, t] = idx // S
            states = idx % S
            if return_states:
                states_out[:, t + 1] = states

        if return_states:
            return emissions, states_out
        return emissions

    def sample(self, length: int, rng: np.random.Generator, **kwargs):
        """Convenience wrapper for a single sequence (see ``sample_batch``)."""
        out = self.sample_batch(1, length, rng, **kwargs)
        if isinstance(out, tuple):
            em, st = out
            return em[0], st[0]
        return out[0]

    # ------------------------------------------------------------------ #
    # Batched belief filtering (for analysis: belief clouds, entropy rate)
    # ------------------------------------------------------------------ #
    def belief_trajectory(self, emissions: np.ndarray, start=None) -> np.ndarray:
        """Optimal beliefs along sequences, vectorised across the batch.

        Parameters
        ----------
        emissions:
            Int array of shape ``[n_seqs, length]`` (a single ``[length]``
            sequence is also accepted and promoted to a batch of 1).
        start:
            Initial belief (default: stationary prior), applied to every sequence.

        Returns
        -------
        beliefs ``[n_seqs, length, n_states]`` where ``beliefs[b, t]`` is the
        posterior belief *after* observing ``emissions[b, :t+1]``. This is the
        sufficient statistic aligned with a causal transformer's residual stream
        at position ``t`` (which is computed from tokens ``0..t`` and used to
        predict token ``t+1``).
        """
        em = np.atleast_2d(np.asarray(emissions, dtype=np.int64))
        B, L = em.shape
        b0 = self.stationary() if start is None else np.asarray(start, dtype=np.float64)
        b = np.tile(b0, (B, 1))  # [B, S]
        out = np.empty((B, L, self.n_states), dtype=np.float64)
        for t in range(L):
            Tx = self.T[em[:, t]]                       # [B, S, S]
            b = np.einsum("bi,bij->bj", b, Tx)          # unnormalised posterior
            b = b / b.sum(axis=1, keepdims=True)
            out[:, t] = b
        return out

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    def __repr__(self):
        return (
            f"{self.name}(n_states={self.n_states}, n_symbols={self.n_symbols}, "
            f"symbols={self.symbol_names})"
        )
