"""Unit tests for the HMM processes and the belief update.

These pin down the *ground truth* before any ML is involved: the transition
tensors, the stationary distribution, and the optimal-observer belief update.
A hand-computed example on a trivial 2-state HMM guards the update direction and
normalisation -- the single easiest thing to get subtly wrong.
"""

import numpy as np
import pytest

from hmms import Mess3, RRXOR, Process
from hmms.mess3 import mess3_tensor, PAPER_X, PAPER_ALPHA


# --------------------------------------------------------------------- #
# A trivial 2-state / 2-symbol HMM with hand-checkable arithmetic.
#   full transition matrix = [[0.9, 0.1], [0.1, 0.9]]  ->  stationary [0.5, 0.5]
# --------------------------------------------------------------------- #
TRIVIAL_T = np.array(
    [
        # symbol 0:  T[0][i, j] = P(next j, emit 0 | i)
        [[0.8, 0.0],
         [0.1, 0.1]],
        # symbol 1:
        [[0.1, 0.1],
         [0.0, 0.8]],
    ]
)


@pytest.fixture
def trivial():
    return Process(TRIVIAL_T, symbol_names=["0", "1"], name="Trivial2State")


# ===================== row-stochasticity ============================= #
def test_full_transition_is_row_stochastic():
    for proc in (Mess3(), RRXOR(), Process(TRIVIAL_T)):
        rows = proc.transition_matrix.sum(axis=1)
        assert np.allclose(rows, 1.0), f"{proc.name} rows={rows}"
        assert np.all(proc.T >= -1e-12)


def test_invalid_tensor_rejected():
    bad = TRIVIAL_T.copy()
    bad[0, 0, 0] += 0.5  # break row-stochasticity
    with pytest.raises(ValueError):
        Process(bad)


# ===================== stationary distribution ======================= #
def test_stationary_is_fixed_point():
    for proc in (Mess3(), RRXOR(), Process(TRIVIAL_T)):
        pi = proc.stationary()
        assert np.allclose(pi.sum(), 1.0)
        assert np.all(pi >= -1e-12)
        # pi @ T_full == pi
        assert np.allclose(pi @ proc.transition_matrix, pi, atol=1e-10)


def test_stationary_known_values():
    assert np.allclose(Process(TRIVIAL_T).stationary(), [0.5, 0.5])
    # Mess3 full T is doubly stochastic and symmetric -> uniform stationary.
    assert np.allclose(Mess3().stationary(), [1 / 3, 1 / 3, 1 / 3])
    # RRXOR: state 0 every 3 steps; the other four share the remaining mass.
    assert np.allclose(RRXOR().stationary(), [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6])


# ===================== belief update (hand-checked) ================== #
def test_belief_update_direction_and_normalisation(trivial):
    # Start certain we are in state 0.
    b0 = np.array([1.0, 0.0])

    # Observing symbol 0 from state 0 keeps us in state 0 (deterministic edge).
    assert np.allclose(trivial.belief_update(b0, 0), [1.0, 0.0])

    # Observing symbol 1 from state 0: b' = [1,0] @ [[0.1,0.1],[0,0.8]] = [0.1,0.1]
    #   -> normalised [0.5, 0.5].
    assert np.allclose(trivial.belief_update(b0, 1), [0.5, 0.5])


def test_belief_after_two_symbol_word(trivial):
    # From [1,0], observe word [1, 1]:
    #   after first 1 -> [0.5, 0.5]
    #   after second 1: [0.5,0.5]@[[0.1,0.1],[0,0.8]] = [0.05, 0.45] -> [0.1, 0.9]
    b = trivial.word_to_belief([1, 1], start=[1.0, 0.0])
    assert np.allclose(b, [0.1, 0.9])


def test_belief_update_zero_probability_symbol_raises(trivial):
    # From state 1 (certain), symbol 0 has probability 0.2 -> fine; construct a
    # genuinely impossible case with a fresh deterministic process instead.
    det = np.array([[[1.0, 0.0], [0.0, 0.0]],   # symbol 0 only from state 0
                    [[0.0, 0.0], [0.0, 1.0]]])  # symbol 1 only from state 1
    proc = Process(det)
    with pytest.raises(ValueError):
        proc.belief_update(np.array([1.0, 0.0]), 1)  # symbol 1 impossible in state 0


def test_next_symbol_dist(trivial):
    b0 = np.array([1.0, 0.0])
    # P(0|state0)=0.8, P(1|state0)=0.2
    assert np.allclose(trivial.next_symbol_dist(b0), [0.8, 0.2])
    # next_symbol_dist always a valid distribution
    assert np.allclose(trivial.next_symbol_dist([0.5, 0.5]).sum(), 1.0)


# ===================== Mess3 specifics =============================== #
def test_mess3_matches_paper_matrix():
    """Pin Mess3 to the exact T^(A) printed in arXiv:2405.15943 Appendix A.1."""
    T = mess3_tensor(PAPER_X, PAPER_ALPHA)
    paper_TA = np.array(
        [[0.765, 0.00375, 0.00375],
         [0.0425, 0.0675, 0.00375],
         [0.0425, 0.00375, 0.0675]]
    )
    assert np.allclose(T[0], paper_TA), "Mess3 T^(A) does not match the paper."
    # Full transition matrix should be the symmetric [[.9,.05,.05],...].
    full = T.sum(axis=0)
    expected = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    assert np.allclose(full, expected)


def test_mess3_default_is_paper_not_library_default():
    """Guard against silently using the epsilon-transformers default x=0.15,a=0.6."""
    assert (PAPER_X, PAPER_ALPHA) == (0.05, 0.85)
    lib_default = mess3_tensor(0.15, 0.6)
    assert not np.allclose(lib_default[0, 0, 0], 0.765)


# ===================== RRXOR specifics ============================== #
def test_rrxor_xor_semantics():
    """The third symbol of each block is the XOR of the two random bits."""
    rng = np.random.default_rng(0)
    # Start every sequence from the block-start state (0) so blocks are aligned.
    em, st = RRXOR().sample_batch(
        2000, 3, rng, init_states=np.zeros(2000, dtype=int), return_states=True
    )
    r1, r2, third = em[:, 0], em[:, 1], em[:, 2]
    assert np.array_equal(third, r1 ^ r2)
    # First two bits are fair coins.
    assert abs(r1.mean() - 0.5) < 0.05
    assert abs(r2.mean() - 0.5) < 0.05


# ===================== sampling reproducibility ===================== #
def test_sampling_is_reproducible():
    a = Mess3().sample_batch(8, 10, np.random.default_rng(123))
    b = Mess3().sample_batch(8, 10, np.random.default_rng(123))
    assert np.array_equal(a, b)


def test_sampling_marginal_matches_stationary_next_symbol():
    """Empirical symbol frequencies match the optimal marginal next-symbol dist."""
    rng = np.random.default_rng(1)
    proc = Mess3()
    em = proc.sample_batch(500, 200, rng)
    freq = np.bincount(em.reshape(-1), minlength=proc.n_symbols) / em.size
    # By symmetry Mess3 emits A/B/C uniformly.
    assert np.allclose(freq, [1 / 3, 1 / 3, 1 / 3], atol=0.02)
