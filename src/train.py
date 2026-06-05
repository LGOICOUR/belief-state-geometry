"""Training: next-token prediction on HMM sequences, to convergence.

The transformer is trained on *exactly one* objective -- predict the next token --
with no knowledge of hidden states or beliefs. The whole result of the paper is
that the belief geometry then appears in the residual stream anyway.

We log the cross-entropy loss against the analytic **optimal in-context loss**
(the information-theoretic floor for a predictor limited to ``n_ctx`` tokens; see
``beliefs.optimal_in_context_loss``). Convergence == loss approaching that floor.

Two presets:

* ``TrainConfig.paper()``  -- the paper's exact recipe: SGD, lr=0.01, batch 64,
  1,000,000 steps, no weight decay.
* ``TrainConfig.fast()``   -- a laptop-friendly recipe (Adam, larger batch, far
  fewer steps) that reaches the same loss floor in minutes. This is the default
  used by the notebooks; the deviation from the paper optimiser is documented in
  the README. Belief recovery is a property of the *converged* model, not of the
  optimiser that got it there -- the probe R^2 is what we ultimately check.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

import beliefs as B
from data import get_device, seed_everything, sample_tokens
from hmms import Mess3, RRXOR
from hmms.base import Process
from model import ArchConfig, build_model, n_params

PROCESSES = {"mess3": Mess3, "rrxor": RRXOR}

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = REPO_ROOT / "checkpoints"
RESULTS_DIR = REPO_ROOT / "results"


# ====================================================================== #
# Config
# ====================================================================== #
@dataclass
class TrainConfig:
    optimizer: str = "adam"          # 'adam' | 'sgd'
    lr: float = 1e-3
    batch_size: int = 1024
    seq_len: int = 10                # == n_ctx
    n_steps: int = 20_000
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0
    seed: int = 0
    device: str | None = None
    log_every: int = 250
    eval_every: int = 1000

    @classmethod
    def paper(cls, **ov) -> "TrainConfig":
        """The paper's exact recipe (arXiv:2405.15943, App. A.4)."""
        cfg = cls(optimizer="sgd", lr=0.01, batch_size=64, n_steps=1_000_000,
                  weight_decay=0.0, grad_clip=None, log_every=2000, eval_every=20_000)
        for k, v in ov.items():
            setattr(cfg, k, v)
        return cfg

    @classmethod
    def fast(cls, **ov) -> "TrainConfig":
        """Laptop-friendly recipe: Adam + big batch + few steps to the same floor."""
        cfg = cls(optimizer="adam", lr=1e-3, batch_size=1024, n_steps=20_000,
                  weight_decay=0.0, grad_clip=1.0, log_every=250, eval_every=1000)
        for k, v in ov.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class TrainResult:
    model: torch.nn.Module
    history: dict           # {'step':[], 'loss':[], 'eval_step':[], 'eval_loss':[]}
    floor_nats: float       # optimal in-context loss
    entropy_rate_bits: float
    final_loss: float
    arch: ArchConfig
    train_cfg: TrainConfig
    process_name: str


# ====================================================================== #
# Training
# ====================================================================== #
def _make_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r}")


@torch.no_grad()
def _eval_loss(model, process, cfg, device, n_seqs=4096, init_states_fn=None) -> float:
    """Mean next-token loss on a fresh (held-out) batch."""
    model.eval()
    rng = np.random.default_rng(20240524)  # fixed eval seed, disjoint draws
    init = init_states_fn(n_seqs, rng) if init_states_fn is not None else None
    tokens = sample_tokens(process, n_seqs, cfg.seq_len, rng, device, init_states=init)
    loss = model(tokens, return_type="loss").item()
    model.train()
    return loss


def train(
    process: Process,
    arch: ArchConfig | None = None,
    train_cfg: TrainConfig | None = None,
    verbose: bool = True,
    process_name: str | None = None,
    init_states_fn=None,
) -> TrainResult:
    """Train next-token prediction. ``init_states_fn(n_seqs, rng) -> [n_seqs]`` lets
    a process seed each batch's hidden states (the mixture uses it for epoch-aligned
    sampling); default None seeds from the stationary distribution."""
    train_cfg = train_cfg or TrainConfig.fast()
    arch = arch or ArchConfig.paper(process, seed=train_cfg.seed)
    device = get_device(train_cfg.device)
    rng = seed_everything(train_cfg.seed)

    model = build_model(arch, device)
    opt = _make_optimizer(model, train_cfg)

    floor = B.optimal_in_context_loss(
        process, np.random.default_rng(train_cfg.seed + 999), arch.n_ctx, units="nats"
    )
    h_bits = B.entropy_rate(process, rng=np.random.default_rng(7), units="bits")

    if verbose:
        pname = process_name or process.name
        print(f"[train] process={pname} device={device} params={n_params(model):,}")
        print(f"[train] optimizer={train_cfg.optimizer} lr={train_cfg.lr} "
              f"batch={train_cfg.batch_size} steps={train_cfg.n_steps}")
        print(f"[train] optimal in-context loss (floor) = {floor:.4f} nats "
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
                gap = avg - floor
                rate = step / (time.time() - t0)
                print(f"  step {step:>7d} | loss {avg:.4f} nats | gap to floor {gap:+.4f} "
                      f"| {rate:.0f} it/s")
        if train_cfg.eval_every and step % train_cfg.eval_every == 0:
            el = _eval_loss(model, process, train_cfg, device, init_states_fn=init_states_fn)
            history["eval_step"].append(step)
            history["eval_loss"].append(el)

    final = _eval_loss(model, process, train_cfg, device, init_states_fn=init_states_fn)
    if verbose:
        print(f"[train] done in {time.time()-t0:.1f}s | final held-out loss "
              f"{final:.4f} nats (floor {floor:.4f}, gap {final-floor:+.4f})")

    return TrainResult(
        model=model, history=history, floor_nats=floor, entropy_rate_bits=h_bits,
        final_loss=final, arch=arch, train_cfg=train_cfg,
        process_name=process_name or process.name,
    )


# ====================================================================== #
# Checkpointing
# ====================================================================== #
def save_checkpoint(result: TrainResult, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "arch": result.arch.as_dict(),
            "train_cfg": asdict(result.train_cfg),
            "process_name": result.process_name,
            "history": result.history,
            "floor_nats": result.floor_nats,
            "entropy_rate_bits": result.entropy_rate_bits,
            "final_loss": result.final_loss,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device | None = None):
    """Rebuild a model from a checkpoint. Returns ``(model, metadata_dict)``."""
    device = get_device(device if isinstance(device, str) or device is None else str(device))
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ArchConfig(**ckpt["arch"])
    model = build_model(arch, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def save_history_json(result: TrainResult, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "process": result.process_name,
        "floor_nats": result.floor_nats,
        "entropy_rate_bits": result.entropy_rate_bits,
        "final_loss_nats": result.final_loss,
        "history": result.history,
        "train_cfg": asdict(result.train_cfg),
        "arch": result.arch.as_dict(),
    }
    path.write_text(json.dumps(payload, indent=2))


# ====================================================================== #
# CLI
# ====================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Train a tiny transformer on an HMM process.")
    ap.add_argument("--process", choices=list(PROCESSES), default="mess3")
    ap.add_argument("--preset", choices=["fast", "paper"], default="fast")
    ap.add_argument("--steps", type=int, default=None, help="override n_steps")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="checkpoint path")
    args = ap.parse_args()

    process = PROCESSES[args.process]()
    cfg = TrainConfig.paper() if args.preset == "paper" else TrainConfig.fast()
    cfg.seed = args.seed
    if args.steps is not None:
        cfg.n_steps = args.steps
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.device is not None:
        cfg.device = args.device

    result = train(process, train_cfg=cfg, process_name=args.process)

    out = Path(args.out) if args.out else CKPT_DIR / f"{args.process}_{args.preset}.pt"
    save_checkpoint(result, out)
    save_history_json(result, RESULTS_DIR / f"{args.process}_train_history.json")
    print(f"[train] saved checkpoint -> {out}")
    print(f"[train] saved history    -> {RESULTS_DIR / f'{args.process}_train_history.json'}")


if __name__ == "__main__":
    main()
