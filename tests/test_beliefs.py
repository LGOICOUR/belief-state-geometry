"""Unit tests for the analytic belief-geometry layer (beliefs.py)."""

import numpy as np
import pytest

from hmms import Mess3, RRXOR
import beliefs as B


# ===================== MSP enumeration ============================== #
def test_rrxor_msp_is_finite_and_valid():
    msp = B.enumerate_msp(RRXOR())
    assert not msp.truncated, "RRXOR MSP should enumerate fully (it is finite)."
    # Every MSP belief is a valid distribution.
    assert np.allclose(msp.beliefs.sum(axis=1), 1.0)
    assert np.all(msp.beliefs >= -1e-12)
    # Every emission row is a valid distribution.
    assert np.allclose(msp.emission_probs.sum(axis=1), 1.0)


def test_rrxor_msp_state_count():
    """The paper reports 36 belief states for RRXOR -- the total reachable set."""
    msp = B.enumerate_msp(RRXOR())
    assert len(msp) == 36, f"expected 36 belief states, got {len(msp)}"


def test_rrxor_synchronises_to_hidden_state_deltas():
    """RRXOR is unifilar, so it synchronises: the *recurrent* belief states are
    exactly the 5 deltas on hidden states (the 36 total are the transient
    synchronisation tree leading into them). This is why RRXOR's interesting
    geometry is the transient structure, distributed across layers."""
    msp = B.enumerate_msp(RRXOR())
    pi = msp.stationary()
    recurrent = msp.beliefs[pi > 1e-9]
    assert len(recurrent) == 5
    # each recurrent belief is a one-hot (a delta on a single hidden state)
    assert np.allclose(np.sort(recurrent, axis=1)[:, -1], 1.0)


def test_mess3_msp_is_infinite():
    """Mess3's MSP is a fractal: depth-limited enumeration keeps growing."""
    shallow = B.enumerate_msp(Mess3(), max_depth=4)
    deeper = B.enumerate_msp(Mess3(), max_depth=8)
    assert len(deeper) > len(shallow)
    # With a tight state cap it reports truncation.
    capped = B.enumerate_msp(Mess3(), max_states=200)
    assert capped.truncated


# ===================== entropy rate ================================= #
def test_rrxor_entropy_rate_is_two_thirds_bits():
    """RRXOR emits 2 fair bits + 1 deterministic bit per 3 symbols => 2/3 bit/symbol."""
    h = B.entropy_rate(RRXOR(), units="bits")
    assert np.isclose(h, 2.0 / 3.0, atol=1e-6), h


def test_entropy_rate_units():
    h_bits = B.entropy_rate(RRXOR(), units="bits")
    h_nats = B.entropy_rate(RRXOR(), units="nats")
    assert np.isclose(h_nats, h_bits * np.log(2), atol=1e-8)


def test_mess3_entropy_rate_mc_is_sane_and_stable():
    h1 = B.entropy_rate_mc(Mess3(), np.random.default_rng(0), units="bits")
    h2 = B.entropy_rate_mc(Mess3(), np.random.default_rng(1), units="bits")
    assert 0.0 < h1 < np.log2(3)            # between deterministic and uniform
    assert abs(h1 - h2) < 0.02              # stable across seeds


def test_entropy_rate_auto_dispatch():
    # Finite MSP -> exact (no rng needed).
    assert np.isclose(B.entropy_rate(RRXOR()), 2 / 3, atol=1e-6)
    # Infinite MSP -> needs rng, else informative error.
    with pytest.raises(ValueError):
        B.entropy_rate(Mess3(), max_states=200)  # truncates, no rng
    val = B.entropy_rate(Mess3(), rng=np.random.default_rng(0), max_states=200)
    assert 0 < val < np.log2(3)


def test_in_context_loss_above_entropy_rate():
    """Finite-context optimal loss is >= the asymptotic entropy rate (same units)."""
    rng = np.random.default_rng(0)
    proc = RRXOR()
    h_nats = B.entropy_rate(proc, units="nats")
    loss = B.optimal_in_context_loss(proc, rng, context_len=10, units="nats")
    assert loss >= h_nats - 1e-3
    assert loss < np.log(proc.n_symbols)  # below the uniform-guess loss


# ===================== sampling belief clouds ======================= #
def test_sample_belief_cloud_shapes():
    bel, nxt = B.sample_belief_cloud(Mess3(), np.random.default_rng(0),
                                     n_seqs=10, length=20)
    assert bel.shape == (200, 3)
    assert nxt.shape == (200,)
    assert np.allclose(bel.sum(axis=1), 1.0)


# ===================== simplex projection / color =================== #
def test_project_to_simplex_2d_vertices():
    verts = B.project_to_simplex_2d(np.eye(3))
    expected = B.simplex_triangle_vertices()
    assert np.allclose(verts, expected)
    # Barycenter maps to the triangle centroid.
    center = B.project_to_simplex_2d(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(center, expected.mean(axis=0))


def test_belief_to_rgb_range():
    rgb = B.belief_to_rgb(np.array([[1, 0, 0], [0.2, 0.3, 0.5]]))
    assert rgb.shape == (2, 3)
    assert np.all((rgb >= 0) & (rgb <= 1))
