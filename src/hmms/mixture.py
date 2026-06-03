"""PHASE 2 STUB -- meta-process mixtures of two HMMs. *Interface only; no behaviour.*

This module is intentionally not implemented in Phase 1. It exists now so that the
rest of the codebase can be written against the abstractions Phase 2 will need,
and so reviewers can see the planned extension is a drop-in, not a rewrite.

Phase-2 experiment (out of scope for this build)
-----------------------------------------------
Combine two sub-processes A and B into a single meta-process whose hidden-state
space is the tagged union ``{(A, s) : s in A.states} ∪ {(B, s) : s in B.states}``.
The generator label (A vs B) is itself a hidden variable the optimal observer
must infer. Construct *pairs* (A, B) that are statistically identical for the
first ``i`` symbols of look-ahead and only diverge afterwards
(``i in {1, 2, 5, 10, 20}``). The question: does the transformer's residual
stream *collapse* mechanistically-distinct generators onto the same belief-space
point until predictive equivalence breaks at horizon ``i``?

Because ``Mixture`` is a ``Process`` like any other, the existing
``data.sample_*`` pipeline and the existing belief machinery (``belief_update``,
``belief_trajectory``, MSP enumeration) all work on it unchanged. The only new
piece is a *probe target*: instead of (or in addition to) regressing residuals
onto the full belief, regress onto the **marginal belief over the generator
label** -- the "which generator" target -- via ``which_generator_target`` below.

Key design point that makes this a drop-in
------------------------------------------
A ``Mixture`` builds a block transition tensor of shape
``[n_symbols, nA + nB, nA + nB]``. If the two generators never switch into each
other, the blocks are independent under the dynamics, but the *observer's belief*
still spreads mass across both blocks and concentrates over time -- which is the
phenomenon under study. Marginalising a meta-belief over the two blocks yields
``P(generator = A | history)``.
"""

from __future__ import annotations

import numpy as np

from .base import Process

_NOT_IMPLEMENTED = (
    "hmms.mixture is a Phase-2 stub: interface only, no behaviour in this build. "
    "See the module docstring for the planned design."
)


class Mixture(Process):
    """PHASE 2 STUB. Meta-process over two sub-processes A and B.

    Intended signature (subject to Phase-2 refinement)::

        Mixture(process_a, process_b, prior_a=0.5, allow_switching=False)

    where ``prior_a`` is the observer's prior probability that the generator is A.
    Constructs a block transition tensor and hands it to ``Process.__init__``, so
    every inherited belief method then works without modification.
    """

    def __init__(self, process_a: Process, process_b: Process, prior_a: float = 0.5,
                 allow_switching: bool = False):
        raise NotImplementedError(_NOT_IMPLEMENTED)


def which_generator_target(meta_belief: np.ndarray, n_states_a: int) -> np.ndarray:
    """PHASE 2 STUB. Marginalise a meta-belief to ``[P(gen=A), P(gen=B)]``.

    The first ``n_states_a`` entries of ``meta_belief`` correspond to generator A.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED)
