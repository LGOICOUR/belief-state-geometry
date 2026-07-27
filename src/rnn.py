"""Phase 3: a minimal recurrent model with a *bottlenecked carried state*.

Where the transformer can re-read the spent coin from its context window, a
recurrent model has no attention: at position ``t`` it sees only the current
token plus a fixed-size hidden state ``h_t`` it has carried forward. Anything
from the past must survive in ``h_t`` (dim ``d_hidden``) -- that is the memory
bottleneck, and it is the whole point of Phase 3: retaining a predictively-defunct
latent now *costs* hidden dimensions, so an optimal predictor has a reason to
compress it away (unlike the transformer, which retains it for free).

``RecurrentLM`` deliberately mirrors the transformer's interfaces:

* ``model(tokens, return_type="loss")`` -- same next-token cross-entropy as
  ``HookedTransformer``, so the analytic loss-floor comparison in ``train.py`` is
  apples-to-apples.
* ``hidden_states(tokens)`` -- returns ``h_t`` at every position, the recurrent
  analogue of caching ``blocks.{l}.hook_resid_post``. The Phase-1/2 probes run on
  it unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import beliefs as B
from data import get_device, sample_tokens, seed_everything
from train import TrainConfig  # reuse the shared training config (optimiser/lr/steps/...)

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"


# ====================================================================== #
# Config + model
# ====================================================================== #
@dataclass
class RNNConfig:
    """Recurrent-model hyperparameters. ``d_hidden`` is the bottleneck we sweep."""

    d_vocab: int
    d_hidden: int = 64
    d_embed: int = 64
    n_layers: int = 1
    cell: str = "gru"          # 'gru' | 'lstm'
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class RecurrentLM(nn.Module):
    """Token embedding -> GRU/LSTM -> unembed, trained on next-token prediction.

    The carried state ``h_t`` (the top layer's hidden state after consuming token
    ``t``) is the sufficient-statistic bottleneck we probe, exactly as Phase 1/2
    probed the residual stream.
    """

    def __init__(self, cfg: RNNConfig):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.d_vocab, cfg.d_embed)
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[cfg.cell]
        self.rnn = rnn_cls(cfg.d_embed, cfg.d_hidden, num_layers=cfg.n_layers, batch_first=True)
        self.unembed = nn.Linear(cfg.d_hidden, cfg.d_vocab)

    def _run(self, tokens: torch.Tensor) -> torch.Tensor:
        e = self.embed(tokens)          # [B, L, d_embed]
        h_seq, _ = self.rnn(e)          # [B, L, d_hidden] -- top-layer h_t per step
        return h_seq

    def forward(self, tokens: torch.Tensor, return_type: str | None = "loss"):
        h_seq = self._run(tokens)
        logits = self.unembed(h_seq)    # [B, L, d_vocab]
        if return_type == "logits":
            return logits
        if return_type is None:
            return None
        # Next-token CE: predict token[t+1] from position t (t = 0 .. L-2), mean over
        # batch*positions -- identical to HookedTransformer(return_type="loss").
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
        return loss

    @torch.no_grad()
    def hidden_states(self, tokens: torch.Tensor, batch_size: int = 4096) -> np.ndarray:
        """Carried state ``h_t`` at every position: ``[n_seqs, seq_len, d_hidden]`` (numpy).

        The recurrent analogue of ``probe.cache_residuals`` -- feed the result to the
        same position-resolved probes.
        """
        self.eval()
        out = []
        for i in range(0, tokens.shape[0], batch_size):
            out.append(self._run(tokens[i : i + batch_size]).cpu().numpy())
        return np.concatenate(out, axis=0)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def probe_hidden_by_position(model, fit_tokens, fit_y, test_tokens, test_y):
    """Decode a per-sequence binary label from the carried state ``h_t`` at each position.

    The recurrent analogue of ``probe.probe_label_by_position`` (which reads ``resid_post``):
    fit an independent logistic probe on ``h_t`` at each position and return held-out
    accuracy vs. position. For the retention probe we decode the *first* epoch's coin
    across the whole sequence -- accuracy in epoch 2 measures whether the GRU *carried*
    the (now-defunct) coin, since it cannot re-read the indicator tokens.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Hf = model.hidden_states(fit_tokens)   # [N, L, d_hidden]
    Ht = model.hidden_states(test_tokens)
    L = Hf.shape[1]
    acc = np.empty(L)
    for p in range(L):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(Hf[:, p, :], fit_y)
        acc[p] = accuracy_score(test_y, clf.predict(Ht[:, p, :]))
    return np.arange(L), acc


# ====================================================================== #
# Training (mirrors train.train, for a carried-state model)
# ====================================================================== #
@dataclass
class RNNTrainResult:
    model: nn.Module
    history: dict
    floor_nats: float
    entropy_rate_bits: float
    final_loss: float
    rnn_cfg: RNNConfig
    train_cfg: TrainConfig
    process_name: str


@torch.no_grad()
def _eval_loss(model, process, cfg, device, n_seqs=4096, init_states_fn=None) -> float:
    model.eval()
    rng = np.random.default_rng(20240524)  # fixed eval seed, disjoint draws
    init = init_states_fn(n_seqs, rng) if init_states_fn is not None else None
    tokens = sample_tokens(process, n_seqs, cfg.seq_len, rng, device, init_states=init)
    loss = model(tokens, return_type="loss").item()
    model.train()
    return loss


def _aligned_floor_nats(process, seq_len, seed=999, n=30000) -> float:
    """Epoch-aligned optimal in-context loss (nats) -- the apt floor for the mixture.

    Mirrors ``experiments._mixture_floor_nats``: seed at epoch boundaries, use the
    epoch-aligned prior, and take the optimal predictor's mean next-token entropy. This
    is *lower* than the stationary ``optimal_in_context_loss`` (which also pays for phase
    uncertainty), and it is the floor the epoch-aligned model is actually trained against,
    so the transformer/GRU comparison stays apples-to-apples.
    """
    rng = np.random.default_rng(seed)
    init = process.aligned_init_states(n, rng)
    em = process.sample_batch(n, seq_len, rng, init_states=init)
    bel = process.belief_trajectory(em, start=process.epoch_start_belief())
    nd = np.einsum("bli,xij->blx", bel, process.T)        # P(next | belief)
    H = B.entropy(nd[:, :-1, :], axis=2, units="nats")    # predict t+1 from belief after t
    return float(H.mean())


def train_rnn(
    process,
    rnn_cfg: RNNConfig,
    train_cfg: TrainConfig | None = None,
    init_states_fn=None,
    verbose: bool = True,
    process_name: str | None = None,
) -> RNNTrainResult:
    """Train the recurrent LM on next-token prediction, mirroring ``train.train``
    (same optimiser/loop/floor logging) but for a carried-state model. For the
    mixture pass ``init_states_fn=process.aligned_init_states`` (epoch-aligned)."""
    train_cfg = train_cfg or TrainConfig.fast()
    device = get_device(train_cfg.device)
    rng = seed_everything(train_cfg.seed)

    model = RecurrentLM(rnn_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    # Epoch-aligned processes (the mixture) are trained/evaluated from epoch boundaries,
    # so the apt floor is the epoch-aligned optimal loss, not the stationary one.
    if init_states_fn is not None and hasattr(process, "epoch_start_belief"):
        floor = _aligned_floor_nats(process, train_cfg.seq_len, seed=train_cfg.seed + 999)
    else:
        floor = B.optimal_in_context_loss(
            process, np.random.default_rng(train_cfg.seed + 999), train_cfg.seq_len, units="nats"
        )
    h_bits = B.entropy_rate(process, rng=np.random.default_rng(7), units="bits")

    if verbose:
        pname = process_name or process.name
        print(f"[rnn] process={pname} device={device} cell={rnn_cfg.cell} "
              f"d_hidden={rnn_cfg.d_hidden} params={n_params(model):,}")
        print(f"[rnn] optimal in-context loss (floor) = {floor:.4f} nats "
              f"| entropy rate = {h_bits:.4f} bits")

    history = {"step": [], "loss": [], "eval_step": [], "eval_loss": []}
    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(1, train_cfg.n_steps + 1):
        init = init_states_fn(train_cfg.batch_size, rng) if init_states_fn is not None else None
        tokens = sample_tokens(process, train_cfg.batch_size, train_cfg.seq_len, rng, device,
                               init_states=init)
        loss = model(tokens, return_type="loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        running += loss.item()

        if step % train_cfg.log_every == 0:
            avg = running / train_cfg.log_every
            running = 0.0
            history["step"].append(step)
            history["loss"].append(avg)
            if verbose:
                rate = step / (time.time() - t0)
                print(f"  step {step:>7d} | loss {avg:.4f} nats | gap to floor "
                      f"{avg - floor:+.4f} | {rate:.0f} it/s")
        if train_cfg.eval_every and step % train_cfg.eval_every == 0:
            el = _eval_loss(model, process, train_cfg, device, init_states_fn=init_states_fn)
            history["eval_step"].append(step)
            history["eval_loss"].append(el)

    final = _eval_loss(model, process, train_cfg, device, init_states_fn=init_states_fn)
    if verbose:
        print(f"[rnn] done in {time.time()-t0:.1f}s | final held-out loss {final:.4f} nats "
              f"(floor {floor:.4f}, gap {final-floor:+.4f})")

    return RNNTrainResult(model, history, floor, h_bits, final, rnn_cfg, train_cfg,
                          process_name or process.name)


# ====================================================================== #
# Checkpointing
# ====================================================================== #
def save_rnn_checkpoint(result: RNNTrainResult, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "rnn_cfg": result.rnn_cfg.as_dict(),
            "train_cfg": asdict(result.train_cfg),
            "process_name": result.process_name,
            "history": result.history,
            "floor_nats": result.floor_nats,
            "final_loss": result.final_loss,
        },
        path,
    )


def load_rnn_checkpoint(path: str | Path, device=None):
    device = get_device(device if isinstance(device, str) or device is None else str(device))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = RNNConfig(**ckpt["rnn_cfg"])
    model = RecurrentLM(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
