"""The Mess3 process.

Mess3 is a 3-hidden-state, 3-symbol Hidden Markov Model introduced by Marzen &
Crutchfield ("Nearly Maximally Predictive Features and Their Dimensions", 2017).
It is *nonunifilar*: the emitted symbol does not determine the next hidden state,
so the optimal observer must maintain a genuine distribution over hidden states.
The set of reachable belief states (its Mixed-State Presentation) is an infinite
**fractal** in the 2-simplex -- which is exactly why Shai et al. (2024) use it:
the fractal is a highly non-trivial, falsifiable prediction for what should
appear in the transformer's residual stream.

Parameterisation
----------------
Mess3 has two scalar parameters, ``x`` and ``a`` (alpha). Each hidden state
"prefers" one symbol. Writing ``b = (1 - a) / 2`` and ``y = 1 - 2x``, the four
distinct transition-emission probabilities are::

    ay = a * y   ax = a * x   by = b * y   bx = b * x

and the three symbol-labeled matrices are (rows = current state, cols = next):

    T[0] = [[ay, bx, bx],    T[1] = [[by, ax, bx],    T[2] = [[by, bx, ax],
            [ax, by, bx],            [bx, ay, bx],            [bx, by, ax],
            [ax, bx, by]]            [bx, ax, by]]            [bx, bx, ay]]

This is the cyclic-symmetric structure of Mess3: applying the 3-cycle to both
states and symbols maps T[0] -> T[1] -> T[2].

Parameter values
----------------
**x = 0.05, a = 0.85.** These are the values used in Shai et al. (2024),
recovered exactly from the paper's printed matrix in Appendix A.1:

    T^(A) = [[0.765,  0.00375, 0.00375],
             [0.0425, 0.0675,  0.00375],
             [0.0425, 0.00375, 0.0675 ]]

(Check: ay = 0.85*0.9 = 0.765; ax = 0.85*0.05 = 0.0425; by = 0.075*0.9 = 0.0675;
bx = 0.075*0.05 = 0.00375.) Source: arXiv:2405.15943, App. A.1 / A.4.

NOTE: a popular community default is ``x=0.15, a=0.6`` (the default in the
authors' ``epsilon-transformers`` library). That is *not* the value used for the
paper's Mess3 figures -- ``test_mess3.py`` pins us to the paper's matrix so this
distinction can't silently drift.
"""

from __future__ import annotations

import numpy as np

from .base import Process

# Paper values (arXiv:2405.15943, Appendix A). See module docstring for the
# arithmetic that recovers these from the printed T^(A) matrix.
PAPER_X = 0.05
PAPER_ALPHA = 0.85


def mess3_tensor(x: float = PAPER_X, alpha: float = PAPER_ALPHA) -> np.ndarray:
    """Build the Mess3 labeled-transition tensor ``T[symbol, state, next_state]``."""
    b = (1.0 - alpha) / 2.0
    y = 1.0 - 2.0 * x
    ay, ax, by, bx = alpha * y, alpha * x, b * y, b * x

    T = np.array(
        [
            [[ay, bx, bx],
             [ax, by, bx],
             [ax, bx, by]],
            [[by, ax, bx],
             [bx, ay, bx],
             [bx, ax, by]],
            [[by, bx, ax],
             [bx, by, ax],
             [bx, bx, ay]],
        ],
        dtype=np.float64,
    )
    return T


class Mess3(Process):
    """The Mess3 process (defaults to the paper's x=0.05, alpha=0.85)."""

    def __init__(self, x: float = PAPER_X, alpha: float = PAPER_ALPHA):
        self.x = x
        self.alpha = alpha
        super().__init__(
            mess3_tensor(x, alpha),
            symbol_names=["A", "B", "C"],
            state_names=["0", "1", "2"],
            name=f"Mess3(x={x}, alpha={alpha})",
        )
