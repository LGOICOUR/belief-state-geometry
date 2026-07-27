"""Unit tests for the Phase 3 recurrent model (the memory-bottleneck arm).

Phase 3 swaps the transformer for a model that must *carry* its state, so these
pin down the two properties the experiment depends on before any training claim:

1. the loss is the same next-token cross-entropy the transformer reports, so the
   analytic loss-floor comparison is apples-to-apples;
2. the model is genuinely **causal and recurrent** -- ``h_t`` depends only on
   tokens up to ``t``. If a future token could leak into ``h_t``, "the state had
   to carry it" would be false and the whole bottleneck argument collapses.
"""

import numpy as np
import pytest
import torch

from hmms import MixtureProcess
from rnn import RNNConfig, RecurrentLM, _aligned_floor_nats


@pytest.fixture
def model():
    return RecurrentLM(RNNConfig(d_vocab=2, d_hidden=8, seed=0))


def _tokens(n=16, L=8, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.integers(0, 2, size=(n, L)), dtype=torch.long)


# ===================== shapes / interfaces ========================== #
def test_hidden_states_shape_matches_positions(model):
    tok = _tokens()
    h = model.hidden_states(tok)
    assert h.shape == (16, 8, 8)          # [n_seqs, seq_len, d_hidden]


def test_logits_shape(model):
    logits = model(_tokens(), return_type="logits")
    assert logits.shape == (16, 8, 2)     # [n_seqs, seq_len, d_vocab]


def test_loss_matches_manual_next_token_cross_entropy(model):
    """The reported loss must be next-token CE over positions 0..L-2 -- the same
    quantity HookedTransformer(return_type='loss') gives, or the floor comparison
    would be meaningless."""
    tok = _tokens()
    logits = model(tok, return_type="logits")
    manual = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 2), tok[:, 1:].reshape(-1)
    )
    assert torch.allclose(model(tok, return_type="loss"), manual, atol=1e-6)


# ===================== the bottleneck property ====================== #
def test_hidden_state_is_causal(model):
    """h_t must not depend on tokens after t. This is what makes the recurrent arm
    a *memory* test: unlike the transformer, nothing later can be re-read."""
    tok = _tokens(n=4, L=8, seed=1)
    h_full = model.hidden_states(tok)
    altered = tok.clone()
    altered[:, 5:] = 1 - altered[:, 5:]           # change only the future
    h_alt = model.hidden_states(altered)
    assert np.allclose(h_full[:, :5], h_alt[:, :5], atol=1e-6)


def test_hidden_state_carries_the_past(model):
    """Conversely, h_t *does* depend on earlier tokens -- otherwise the probe would
    be reading nothing and retention would be trivially zero."""
    tok = _tokens(n=4, L=8, seed=2)
    altered = tok.clone()
    altered[:, 0] = 1 - altered[:, 0]             # change only the first token
    h_full, h_alt = model.hidden_states(tok), model.hidden_states(altered)
    assert not np.allclose(h_full[:, -1], h_alt[:, -1], atol=1e-6)


def test_width_controls_carried_state_size():
    """d_hidden is the bottleneck the sweep varies, so it must set the state size."""
    for d in (2, 8, 32):
        h = RecurrentLM(RNNConfig(d_vocab=2, d_hidden=d)).hidden_states(_tokens())
        assert h.shape[-1] == d


# ===================== the analytic floor =========================== #
@pytest.mark.parametrize("horizon,tail,seq_len", [(2, 2, 8), (2, 2, 12), (3, 2, 10)])
def test_aligned_floor_matches_hand_counted_entropy(horizon, tail, seq_len):
    """The epoch-aligned floor is a *finite-window* average, not the asymptotic
    entropy rate, so count it by hand over the actual prediction positions.

    Predicting token ``j`` costs 1 bit if it is unpredictable and 0 if forced:
    prefix tokens are uniform (1 bit); the *first* tail token reveals the fair coin
    (1 bit); every later tail token is determined by the revealed coin (0 bits). So
    ``cost(j) = 1 if (j % epoch_len) <= horizon else 0``, averaged over j = 1..L-1.

    For Mixture(2,2) at seq_len=8 this is 5/7 bit = 0.4951 nats -- the number P0
    convergence is judged against (the asymptotic rate, 3/4 bit, is a different and
    wrong yardstick for a finite context).
    """
    proc = MixtureProcess(horizon=horizon, tail=tail)
    floor = _aligned_floor_nats(proc, seq_len=seq_len, n=4000)
    bits = np.mean([1.0 if (j % proc.epoch_len) <= horizon else 0.0
                    for j in range(1, seq_len)])
    assert floor == pytest.approx(bits * np.log(2), abs=0.02)
