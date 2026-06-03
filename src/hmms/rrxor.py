"""The RRXOR ("Random, Random, XOR") process.

RRXOR emits length-3 blocks: two independent fair random bits ``r1, r2`` followed
by their XOR ``r1 ^ r2``. As a *presented* process it has 5 hidden states. Unlike
Mess3 its Mixed-State Presentation is finite (36 reachable belief states), but it
is the headline example in Shai et al. (2024) for belief geometry being
**distributed across layers** rather than read off the final residual stream
alone.

Hidden-state semantics (matching the matrices in Appendix A.1)
-------------------------------------------------------------
State 0 is the block-start state.

    state 0  --emit r1-->  state 1 (if r1=0)  or  state 2 (if r1=1)      [each p=0.5]
    state 1  --emit r2-->  state 4 (if r2=0)  or  state 3 (if r2=1)      [each p=0.5]
    state 2  --emit r2-->  state 3 (if r2=0)  or  state 4 (if r2=1)      [each p=0.5]
    state 3  --emit 1 (deterministic)-->  state 0
    state 4  --emit 0 (deterministic)-->  state 0

So the third symbol emitted from states {3,4} is exactly XOR(r1, r2):
  (r1,r2)=(0,0) -> state 4 -> emit 0 = 0^0   (1,1) -> state 4 -> emit 0 = 1^1
  (0,1) -> state 3 -> emit 1 = 0^1           (1,0) -> state 3 -> emit 1 = 1^0

Transition matrices (verbatim from arXiv:2405.15943, Appendix A.1), with
``T[x, i, j] = P(next state j, emit x | state i)``:

    T^(0) = [[0, .5, 0,  0,  0 ],      T^(1) = [[0, 0,  .5, 0,  0 ],
             [0, 0,  0,  0,  .5],               [0, 0,  0,  .5, 0 ],
             [0, 0,  0,  .5, 0 ],               [0, 0,  0,  0,  .5],
             [0, 0,  0,  0,  0 ],               [1, 0,  0,  0,  0 ],
             [1, 0,  0,  0,  0 ]]               [0, 0,  0,  0,  0 ]]

Stationary distribution is [1/3, 1/6, 1/6, 1/6, 1/6] (state 0 is visited every
3 steps; the other four are visited every 3 steps collectively). The unit tests
check this.
"""

from __future__ import annotations

import numpy as np

from .base import Process

# T[symbol, state, next_state]; symbol 0 then symbol 1.
RRXOR_TENSOR = np.array(
    [
        # --- symbol 0 ---
        [[0.0, 0.5, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.5],
         [0.0, 0.0, 0.0, 0.5, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0, 0.0]],
        # --- symbol 1 ---
        [[0.0, 0.0, 0.5, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.5, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.5],
         [1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]],
    ],
    dtype=np.float64,
)


class RRXOR(Process):
    """The Random-Random-XOR process (5 hidden states, binary alphabet)."""

    def __init__(self):
        super().__init__(
            RRXOR_TENSOR,
            symbol_names=["0", "1"],
            state_names=["S0", "S1", "S2", "S3", "S4"],
            name="RRXOR",
        )
