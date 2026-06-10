"""Unit tests for the Phase 2 mixture process (belief-state collapse experiment).

These pin down the construction before any ML: the transition tensor, the
collapse -> bifurcation -> reset dynamics of the generator marginal P(Z=A | history),
and the finiteness of the Mixed-State Presentation. The hand-checked belief
trajectory is the key guard -- it is the analytic prediction the Phase 2 probe
will be compared against.
"""

import numpy as np
import pytest

from hmms import MixtureProcess, GEN_A, GEN_B
from beliefs import enumerate_msp, entropy_rate


def epoch_start_belief(proc: MixtureProcess) -> np.ndarray:
    """Belief at an epoch boundary: 1/2 on each generator's first prefix state."""
    b = np.zeros(proc.n_states)
    b[proc.epoch_start_states[GEN_A]] = 0.5
    b[proc.epoch_start_states[GEN_B]] = 0.5
    return b


# ===================== construction / shape ========================= #
@pytest.mark.parametrize("h,tail", [(1, 2), (2, 2), (5, 3), (3, 1)])
def test_state_count_and_row_stochastic(h, tail):
    p = MixtureProcess(h, tail)
    assert p.n_states == 2 * (h + tail)
    assert p.n_symbols == 2
    assert np.allclose(p.transition_matrix.sum(axis=1), 1.0)
    assert np.all(p.T >= -1e-12)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        MixtureProcess(horizon=0, tail=2)
    with pytest.raises(ValueError):
        MixtureProcess(horizon=2, tail=0)


# ===================== symmetry / stationary ======================== #
def test_stationary_uniform_and_balanced():
    p = MixtureProcess(2, 2)
    pi = p.stationary()
    assert np.allclose(pi, 1.0 / p.n_states)          # uniform by symmetry
    assert np.isclose(p.generator_marginal(pi), 0.5)  # A and B equally likely


# ============ the key dynamics: collapse -> bifurcate -> reset ====== #
def test_collapse_bifurcate_reset_hand_checked():
    """The generator marginal is 1/2 through the prefix, commits at horizon+1
    (the first indicator), and resets at the epoch boundary."""
    p = MixtureProcess(horizon=2, tail=2)
    start = epoch_start_belief(p)

    # Z=A epoch: prefix [1,0], tail [0,0]  (A's indicator is 0)
    pa_A = p.generator_marginal(p.belief_trajectory(np.array([1, 0, 0, 0]), start=start)[0])
    assert np.allclose(pa_A, [0.5, 0.5, 1.0, 0.5])

    # Z=B epoch: prefix [1,0], tail [1,1]  (B's indicator is 1)
    pa_B = p.generator_marginal(p.belief_trajectory(np.array([1, 0, 1, 1]), start=start)[0])
    assert np.allclose(pa_B, [0.5, 0.5, 0.0, 0.5])


def test_prefix_is_predictively_uninformative():
    """E4 control: P(Z=A) stays exactly 1/2 through the whole random prefix,
    for any prefix bits -- Z is information-theoretically absent before the reveal."""
    p = MixtureProcess(horizon=3, tail=2)
    start = epoch_start_belief(p)
    for prefix in ([0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]):
        seq = np.array(prefix + [0, 0])  # complete a valid Z=A epoch
        pa = p.generator_marginal(p.belief_trajectory(seq, start=start)[0])
        assert np.allclose(pa[: p.horizon], 0.5), (prefix, pa)
        assert pa[p.horizon] == pytest.approx(1.0)  # commit at the first indicator


# ===================== sampling generates the structure ============= #
def test_sampling_tail_equals_indicator():
    """Seeded at a generator's epoch start, the tail emits that generator's
    indicator deterministically and the prefix bits are fair coins."""
    rng = np.random.default_rng(0)
    p = MixtureProcess(horizon=3, tail=2)
    N = 2000
    for Z, indicator in ((GEN_A, 0), (GEN_B, 1)):
        init = np.full(N, p.epoch_start_states[Z], dtype=int)
        em = p.sample_batch(N, p.epoch_len, rng, init_states=init)
        assert np.all(em[:, p.horizon:] == indicator)      # deterministic tail
        assert abs(em[:, : p.horizon].mean() - 0.5) < 0.05  # uniform prefix


# ===================== MSP / entropy rate =========================== #
def test_msp_finite():
    msp = enumerate_msp(MixtureProcess(2, 2))
    assert not msp.truncated and len(msp) > 0


def test_entropy_rate_matches_information_count():
    """h uniform random bits + 1 bit to reveal Z, per (h + tail) tokens."""
    for h, tail in [(2, 2), (3, 2), (2, 3)]:
        p = MixtureProcess(h, tail)
        assert np.isclose(entropy_rate(p, units="bits"), (h + 1) / (h + tail))
