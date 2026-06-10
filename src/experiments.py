"""End-to-end experiment orchestration.

These functions take a trained checkpoint and produce the deliverables: the probe
R^2 numbers, the figures in ``results/``, and ``results/metrics.json``. The
notebooks are thin wrappers that call these and display the returned figures, so
all real logic is here (and therefore importable and testable, unlike notebook
cells).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import beliefs as B
from data import make_eval_set, make_mixture_eval_set, get_device, sample_tokens
from hmms import Mess3, RRXOR, MixtureProcess
from model import ArchConfig
from probe import (
    ProbeResult,
    cache_residuals,
    flatten_beliefs,
    probe_all_layers,
    probe_concat_layers,
    probe_label_by_position,
    _residuals_3d,
    best_layer,
)
from train import load_checkpoint, train, save_checkpoint, TrainConfig
import viz

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def _rel(path: Path) -> str:
    """Path relative to the repo root, for metrics.json portability."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _measure_loss_floor(model, process, seq_len, device, n_loss=40000, n_floor=40000):
    """Accurate (large-sample) held-out loss and optimal in-context floor, in nats.

    Recomputed here rather than read from the checkpoint: the train-time values are
    small-sample MC estimates and can straddle the true floor by noise, which would
    misleadingly show the model 'below optimal'. These tight estimates report the
    honest relationship (model loss >= floor, both ~ the entropy rate).
    """
    floor = B.optimal_in_context_loss(process, np.random.default_rng(12345), seq_len,
                                      n_seqs=n_floor, units="nats")
    rng = np.random.default_rng(54321)
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(5):
            tok = sample_tokens(process, n_loss // 5, seq_len, rng, device)
            losses.append(model(tok, return_type="loss").item())
    return float(np.mean(losses)), float(floor)


def _predict_on_cloud(model, estimator, process, layer, hook, n_seqs, seq_len, seed, device):
    """Apply a fitted single-layer probe to a fresh large set (for a dense figure)."""
    data = make_eval_set(process, n_seqs, seq_len, seed, device)
    X = cache_residuals(model, data.tokens, hook, [layer])[layer]
    return flatten_beliefs(data), estimator.predict(X)


# ====================================================================== #
# Mess3 (headline)
# ====================================================================== #
def run_mess3_experiment(
    checkpoint_path,
    n_fit: int = 4000,
    n_test: int = 4000,
    n_figure: int = 40000,
    hook: str = "resid_post",
    seed: int = 0,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Probe the Mess3 model, build the headline figure, write metrics.json."""
    device = get_device(device)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    process = Mess3()
    seq_len = ckpt["arch"]["n_ctx"]
    held_loss, floor = _measure_loss_floor(model, process, seq_len, device)

    fit_data = make_eval_set(process, n_fit, seq_len, seed + 1, device)
    test_data = make_eval_set(process, n_test, seq_len, seed + 2, device)
    results = probe_all_layers(model, fit_data, test_data, hook=hook)

    # The paper recovers Mess3 from the *final* residual stream.
    headline_layer = model.cfg.n_layers - 1
    headline = results[headline_layer]
    bl = best_layer(results)

    # Dense cloud for the figure: apply the fitted probe to a large fresh sample.
    yt, yp = _predict_on_cloud(model, headline.estimator, process, headline_layer,
                               hook, n_figure, seq_len, seed + 3, device)
    fig_result = ProbeResult(headline_layer, hook, headline.r2, headline.r2_per_coord,
                             yt, yp, headline.n_fit, headline.n_test, headline.alpha,
                             headline.estimator)

    figures = {}
    fig_head = viz.plot_simplex_comparison(
        fig_result,
        title="Mess3: belief-state geometry recovered from the residual stream",
        save_path=results_dir / "mess3_headline.png" if save else None,
    )
    figures["headline"] = _rel(results_dir / "mess3_headline.png")
    fig_curve = viz.plot_training_curve(
        ckpt["history"], floor, ckpt["entropy_rate_bits"],
        title="Mess3: training loss vs analytic optimal",
        save_path=results_dir / "mess3_training_curve.png" if save else None,
    )
    figures["training_curve"] = _rel(results_dir / "mess3_training_curve.png")
    fig_layers = viz.plot_layer_r2(
        results, title="Mess3: belief recovery R² by layer",
        save_path=results_dir / "mess3_layer_r2.png" if save else None,
    )
    figures["layer_r2"] = _rel(results_dir / "mess3_layer_r2.png")

    metrics = {
        "process": "mess3",
        "seed": seed,
        "model": ckpt["arch"],
        "training": {
            "optimizer": ckpt["train_cfg"]["optimizer"],
            "lr": ckpt["train_cfg"]["lr"],
            "batch_size": ckpt["train_cfg"]["batch_size"],
            "n_steps": ckpt["train_cfg"]["n_steps"],
            "final_loss_nats": held_loss,
            "optimal_in_context_loss_nats": floor,
            "loss_gap_to_floor_nats": held_loss - floor,
            "entropy_rate_bits": ckpt["entropy_rate_bits"],
            "entropy_rate_nats": ckpt["entropy_rate_bits"] * np.log(2),
        },
        "probe": {
            "hook": hook,
            "n_fit_positions": int(headline.n_fit),
            "n_test_positions": int(headline.n_test),
            "headline_layer": headline_layer,
            "r2_headline_layer": headline.r2,
            "r2_per_coord_headline": headline.r2_per_coord.tolist(),
            "r2_per_layer": {str(l): results[l].r2 for l in sorted(results)},
            "best_layer": int(bl),
            "r2_best_layer": results[bl].r2,
        },
        "figures": figures,
    }
    if save:
        _write_metrics(metrics, results_dir / "metrics_mess3.json")

    return {
        "metrics": metrics,
        "results": results,
        "headline": headline,
        "fig_result": fig_result,
        "figures": {"headline": fig_head, "training_curve": fig_curve, "layer_r2": fig_layers},
        "checkpoint": ckpt,
        "model": model,
    }


def _write_metrics(metrics: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def write_combined_metrics(results_dir: Path = RESULTS_DIR) -> dict:
    """Merge per-process metrics into a single ``results/metrics.json``."""
    combined = {}
    for name in ("mess3", "rrxor"):
        p = Path(results_dir) / f"metrics_{name}.json"
        if p.exists():
            combined[name] = json.loads(p.read_text())
    (Path(results_dir) / "metrics.json").write_text(json.dumps(combined, indent=2))
    return combined


# ====================================================================== #
# RRXOR (distributed representation across layers)
# ====================================================================== #
def run_rrxor_experiment(
    checkpoint_path,
    n_fit: int = 8000,
    n_test: int = 8000,
    hook: str = "resid_post",
    alpha: float = 1.0,
    seed: int = 0,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Probe RRXOR per-layer and across-layers; run the future-information analysis.

    We use light Ridge (``alpha=1.0``) rather than plain OLS here. The residual
    stream is *cumulative* (``resid_post[l]`` contains all earlier layers), so the
    across-layer concatenation is highly collinear; plain OLS is then numerically
    unstable and *underperforms* a single layer. With light regularisation the
    distributed-representation result is robust (concat ≫ any single layer for
    every alpha in [1, 1000]). Mess3, recovered cleanly from one layer, needs no
    such regularisation and uses plain OLS.
    """
    device = get_device(device)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    process = RRXOR()
    seq_len = ckpt["arch"]["n_ctx"]
    held_loss, floor = _measure_loss_floor(model, process, seq_len, device)

    fit_data = make_eval_set(process, n_fit, seq_len, seed + 1, device)
    test_data = make_eval_set(process, n_test, seq_len, seed + 2, device)

    per_layer = probe_all_layers(model, fit_data, test_data, hook=hook, alpha=alpha)
    concat = probe_concat_layers(model, fit_data, test_data, hook=hook, alpha=alpha)
    bl = best_layer(per_layer)

    fi = future_information_analysis(process, fit_data, test_data, concat)

    figures = {}
    fig_layers = viz.plot_layer_r2(
        per_layer, concat,
        title="RRXOR: belief recovery R² — distributed across layers",
        save_path=results_dir / "rrxor_layer_r2.png" if save else None,
    )
    figures["layer_r2"] = _rel(results_dir / "rrxor_layer_r2.png")
    fig_scatter = viz.plot_pred_vs_true(
        concat, state_labels=process.state_names,
        title="RRXOR: belief recovered from concatenated layers",
        save_path=results_dir / "rrxor_pred_vs_true.png" if save else None,
    )
    figures["pred_vs_true"] = _rel(results_dir / "rrxor_pred_vs_true.png")
    fig_future = viz.plot_future_information(
        fi, title="RRXOR: belief carries information beyond the next token",
        save_path=results_dir / "rrxor_future_information.png" if save else None,
    )
    figures["future_information"] = _rel(results_dir / "rrxor_future_information.png")

    metrics = {
        "process": "rrxor",
        "seed": seed,
        "model": ckpt["arch"],
        "training": {
            "optimizer": ckpt["train_cfg"]["optimizer"],
            "lr": ckpt["train_cfg"]["lr"],
            "batch_size": ckpt["train_cfg"]["batch_size"],
            "n_steps": ckpt["train_cfg"]["n_steps"],
            "final_loss_nats": held_loss,
            "optimal_in_context_loss_nats": floor,
            "loss_gap_to_floor_nats": held_loss - floor,
            "entropy_rate_bits": ckpt["entropy_rate_bits"],
        },
        "probe": {
            "hook": hook,
            "ridge_alpha": alpha,
            "n_test_positions": int(concat.n_test),
            "r2_per_layer": {str(l): per_layer[l].r2 for l in sorted(per_layer)},
            "best_single_layer": int(bl),
            "r2_best_single_layer": per_layer[bl].r2,
            "r2_concat_all_layers": concat.r2,
            "concat_beats_best_single_layer": bool(concat.r2 > per_layer[bl].r2),
        },
        "future_information": {
            "r2_residual_to_belief": fi["r2_residual_to_belief"],
            "r2_nexttoken_dist_to_belief": fi["r2_nexttoken_to_belief"],
            "interpretation": (
                "The residual stream linearly encodes the belief far better than the "
                "next-token distribution can (a many-to-one function of belief), so it "
                "carries information about the whole future, not just the next token."
            ),
        },
        "figures": figures,
    }
    if save:
        _write_metrics(metrics, results_dir / "metrics_rrxor.json")

    return {
        "metrics": metrics,
        "per_layer": per_layer,
        "concat": concat,
        "future_information": fi,
        "figures": {"layer_r2": fig_layers, "pred_vs_true": fig_scatter,
                    "future_information": fig_future},
        "checkpoint": ckpt,
        "model": model,
    }


# ====================================================================== #
# Secondary analysis: belief carries information beyond the next token
# ====================================================================== #
def future_information_analysis(
    process,
    fit_data,
    test_data,
    concat_result: ProbeResult,
    n_pairs: int = 150_000,
    seed: int = 0,
):
    """Quantify that the recovered belief carries info beyond the next token.

    Two complementary measurements:

    1. **Recoverability gap.** Fit a linear map from the *next-token distribution*
       to the belief (the best you could do if the residual only held next-token
       information) and compare its held-out R^2 to the residual->belief probe.
       The next-token distribution is a many-to-one function of the belief, so it
       cannot reconstruct the belief -- a large gap means the residual holds
       strictly more than next-token information.

    2. **Pairwise distances.** For random pairs of positions, compare belief
       distance to next-token-distribution distance. Pairs with ~equal next-token
       distributions but different beliefs (mass at next-token distance ~0,
       belief distance > 0) are positions the next token cannot tell apart but the
       belief -- and hence the residual stream -- can.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    b_fit = flatten_beliefs(fit_data)
    b_test = concat_result.y_true
    p_fit = np.einsum("ni,xij->nx", b_fit, process.T)    # next-token dist
    p_test = np.einsum("ni,xij->nx", b_test, process.T)

    est = LinearRegression().fit(p_fit, b_fit)
    r2_nexttoken_to_belief = float(r2_score(b_test, est.predict(p_test)))

    rng = np.random.default_rng(seed)
    N = b_test.shape[0]
    i = rng.integers(0, N, n_pairs)
    j = rng.integers(0, N, n_pairs)
    d_belief = np.linalg.norm(b_test[i] - b_test[j], axis=1)
    d_nexttok = np.linalg.norm(p_test[i] - p_test[j], axis=1)

    return {
        "r2_residual_to_belief": concat_result.r2,
        "r2_nexttoken_to_belief": r2_nexttoken_to_belief,
        "d_belief": d_belief,
        "d_nexttok": d_nexttok,
    }


# ====================================================================== #
# Phase 2: mixture process (belief-state collapse / retention)
# ====================================================================== #
def _mixture_floor_nats(process, seq_len, seed=999, n=30000):
    """Epoch-aligned optimal in-context loss (nats): the apt floor for this model.

    Like ``beliefs.optimal_in_context_loss`` but seeded at epoch boundaries and
    using the epoch-aligned prior, matching how the mixture model is trained/evaluated.
    """
    rng = np.random.default_rng(seed)
    init = process.aligned_init_states(n, rng)
    em = process.sample_batch(n, seq_len, rng, init_states=init)
    bel = process.belief_trajectory(em, start=process.epoch_start_belief())
    nd = np.einsum("bli,xij->blx", bel, process.T)        # P(next | belief)
    H = B.entropy(nd[:, :-1, :], axis=2, units="nats")    # predict t+1 from belief after t
    return float(H.mean())


@torch.no_grad()
def _eval_aligned_loss(model, process, seq_len, device, n=8192):
    """Held-out next-token loss (nats) on epoch-aligned sequences."""
    rng = np.random.default_rng(777)
    init = process.aligned_init_states(n, rng)
    tok = sample_tokens(process, n, seq_len, rng, device, init_states=init)
    model.eval()
    return float(model(tok, return_type="loss").item())


def run_mixture_experiment(
    horizon: int = 2,
    tail: int = 2,
    n_ctx: int = 8,
    n_steps: int = 8000,
    n_fit: int = 6000,
    n_test: int = 6000,
    seed: int = 0,
    device=None,
    checkpoint_path=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Smallest end-to-end mixture experiment (retention probe).

    Trains an epoch-aligned mixture model, then decodes the *first* epoch's
    generator label from the residual stream at every position. Accuracy is at
    chance before the reveal, high once the indicator is in the input, and -- in
    epoch >= 2, where that label is neither in the input nor predictively needed --
    measures **retention**: pure predictive sufficiency would discard it (chance),
    extra mechanistic memory would keep it (near the upper bound).
    """
    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = process.epoch_len

    # --- train (or load) ---
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        model, ckpt = load_checkpoint(checkpoint_path, device)
        history = ckpt.get("history", {"step": [], "loss": []})
        ent_bits = ckpt.get("entropy_rate_bits", float("nan"))
    else:
        arch = ArchConfig.paper(process, n_ctx=n_ctx, seed=seed)
        cfg = TrainConfig.fast(seq_len=n_ctx, n_steps=n_steps, seed=seed)
        result = train(process, arch, cfg, init_states_fn=process.aligned_init_states,
                       process_name=f"mixture(h={horizon},tail={tail})")
        model, history, ent_bits = result.model, result.history, result.entropy_rate_bits
        if save:
            out = checkpoint_path or (REPO_ROOT / "checkpoints" / f"mixture_h{horizon}_t{tail}.pt")
            save_checkpoint(result, out)

    floor = _mixture_floor_nats(process, n_ctx, seed=seed + 999)
    held_loss = _eval_aligned_loss(model, process, n_ctx, device)

    # --- retention probe: decode the first epoch's generator label by position ---
    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
    positions, model_acc = probe_label_by_position(
        model, fit.tokens, fit.z_first, test.tokens, test.z_first
    )
    upper_acc = np.array([1.0 if p >= horizon else 0.5 for p in positions])

    prefix = model_acc[:horizon]            # before reveal (control: ~0.5)
    revealed = model_acc[horizon:epoch_len]  # epoch 1 after reveal (~1.0)
    retention = model_acc[epoch_len:]        # epoch >= 2 (the result)
    retention_mean = float(retention.mean()) if retention.size else float("nan")

    figures = {}
    fig_dec = viz.plot_generator_decodability(
        positions, model_acc, upper_acc, horizon, epoch_len,
        title=f"Mixture (horizon={horizon}, tail={tail}): generator label by position",
        save_path=results_dir / "mixture_decodability.png" if save else None,
    )
    figures["decodability"] = _rel(results_dir / "mixture_decodability.png")
    fig_curve = viz.plot_training_curve(
        history, floor, ent_bits, title="Mixture: training loss vs analytic floor",
        save_path=results_dir / "mixture_training_curve.png" if save else None,
    )
    figures["training_curve"] = _rel(results_dir / "mixture_training_curve.png")

    metrics = {
        "process": f"mixture(h={horizon},tail={tail})",
        "seed": seed,
        "config": {"horizon": horizon, "tail": tail, "epoch_len": epoch_len, "n_ctx": n_ctx},
        "training": {
            "final_loss_nats": held_loss,
            "optimal_in_context_loss_nats": floor,
            "loss_gap_to_floor_nats": held_loss - floor,
            "entropy_rate_bits": ent_bits,
        },
        "retention_probe": {
            "accuracy_by_position": [round(float(a), 4) for a in model_acc],
            "upper_bound_by_position": [float(a) for a in upper_acc],
            "mean_prefix_accuracy": float(prefix.mean()) if prefix.size else None,
            "mean_revealed_accuracy": float(revealed.mean()) if revealed.size else None,
            "mean_retention_accuracy": retention_mean,
            "interpretation": (
                "First-epoch generator label decoded from the residual at each position. "
                "Chance (~0.5) before the reveal; ~1.0 once the indicator is in the input. "
                "In epoch >= 2 the label is neither in the current input nor predictively "
                f"needed, so the retention accuracy ({retention_mean:.3f}) is the result: "
                "~0.5 = discards predictively-useless identity (pure predictive sufficiency); "
                "near the upper bound = retains it (extra mechanistic memory)."
            ),
        },
        "figures": figures,
    }
    if save:
        _write_metrics(metrics, results_dir / "metrics_mixture.json")

    return {
        "metrics": metrics,
        "positions": positions,
        "model_acc": model_acc,
        "upper_acc": upper_acc,
        "figures": {"decodability": fig_dec, "training_curve": fig_curve},
        "model": model,
        "process": process,
    }


def run_mixture_sweep(
    i_values=(1, 2, 5, 10, 20),
    seeds=(0, 1, 2),
    tail: int = 2,
    n_steps: int = 8000,
    n_fit: int = 3000,
    n_test: int = 3000,
    converged_tol: float = 0.02,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Horizon x seed sweep of the retention probe (robustness for the claim).

    For each (horizon i, seed): train ``MixtureProcess(i, tail)`` epoch-aligned with
    ``n_ctx = 2*(i+tail)`` (two epochs), confirm the loss reaches the floor, and
    decode the first-epoch coin by position. Records the decodability curve, the
    onset (should track ``i``), and the mean retention accuracy in epoch >= 2.
    Writes results incrementally so a partial run is still usable.
    """
    device = get_device(device)
    records = []
    json_path = results_dir / "mixture_sweep.json"

    for i in i_values:
        n_ctx = 2 * (i + tail)
        epoch_len = i + tail
        for seed in seeds:
            try:
                process = MixtureProcess(i, tail)
                arch = ArchConfig.paper(process, n_ctx=n_ctx, seed=seed)
                cfg = TrainConfig.fast(seq_len=n_ctx, n_steps=n_steps, seed=seed)
                result = train(process, arch, cfg, verbose=False,
                               init_states_fn=process.aligned_init_states,
                               process_name=f"mix(i={i},s={seed})")
                model = result.model
                floor = _mixture_floor_nats(process, n_ctx, seed=seed + 999)
                loss = _eval_aligned_loss(model, process, n_ctx, device)
                fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
                test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
                _, acc = probe_label_by_position(model, fit.tokens, fit.z_first,
                                                 test.tokens, test.z_first)
                prefix, revealed, retention = acc[:i], acc[i:epoch_len], acc[epoch_len:]
                onset = int(np.argmax(acc >= 0.9)) if np.any(acc >= 0.9) else -1
                rec = {
                    "horizon": i, "seed": seed, "tail": tail, "n_ctx": n_ctx,
                    "epoch_len": epoch_len, "loss_nats": round(loss, 4),
                    "floor_nats": round(floor, 4), "gap_nats": round(loss - floor, 4),
                    "converged": bool(loss - floor < converged_tol),
                    "reveal_position": i, "onset_position": onset,
                    "mean_prefix_acc": float(prefix.mean()) if prefix.size else None,
                    "mean_revealed_acc": float(revealed.mean()) if revealed.size else None,
                    "mean_retention_acc": float(retention.mean()) if retention.size else None,
                    "accuracy_by_position": [round(float(a), 4) for a in acc],
                }
                records.append(rec)
                print(f"[sweep] i={i:>2} seed={seed} n_ctx={n_ctx:>2} loss={loss:.4f} "
                      f"gap={loss-floor:+.4f} onset={onset}(reveal={i}) "
                      f"retention={rec['mean_retention_acc']:.3f}", flush=True)
                del model, result, fit, test
            except Exception as e:  # keep the sweep alive; record the failure
                records.append({"horizon": i, "seed": seed, "error": repr(e)})
                print(f"[sweep] i={i} seed={seed} FAILED: {e!r}", flush=True)
            if save:
                _write_metrics({"records": records}, json_path)

    # ---- aggregate over seeds (prefer converged runs) ----
    ok = [r for r in records if "error" not in r]
    by_i, decay_by_i = {}, {}
    for i in i_values:
        rs_all = [r for r in ok if r["horizon"] == i]
        rs = [r for r in rs_all if r["converged"]] or rs_all
        if not rs:
            continue
        ret = np.array([r["mean_retention_acc"] for r in rs])
        by_i[i] = {
            "mean": float(ret.mean()), "std": float(ret.std()), "n_seeds": len(rs),
            "revealed": float(np.mean([r["mean_revealed_acc"] for r in rs])),
            "prefix": float(np.mean([r["mean_prefix_acc"] for r in rs])),
            "onset_mean": float(np.mean([r["onset_position"] for r in rs])),
            "all_converged": all(r["converged"] for r in rs_all),
        }
        accs = np.array([r["accuracy_by_position"] for r in rs])  # [n_seeds, n_ctx]
        mean_acc = accs.mean(axis=0)
        decay_by_i[i] = ((np.arange(len(mean_acc)) - i).tolist(), mean_acc.tolist())

    summary = {
        "config": {"i_values": list(i_values), "seeds": list(seeds), "tail": tail,
                   "n_steps": n_steps, "n_fit": n_fit, "n_test": n_test},
        "by_horizon": by_i,
        "onset_tracks_horizon": {str(i): by_i[i]["onset_mean"] for i in by_i},
        "records": records,
    }
    if save and by_i:
        viz.plot_retention_vs_horizon(
            by_i, save_path=results_dir / "mixture_retention_vs_horizon.png")
        viz.plot_retention_decay(
            {i: (np.array(decay_by_i[i][0]), np.array(decay_by_i[i][1])) for i in decay_by_i},
            save_path=results_dir / "mixture_retention_decay.png")
        summary["figures"] = {
            "retention_vs_horizon": _rel(results_dir / "mixture_retention_vs_horizon.png"),
            "retention_decay": _rel(results_dir / "mixture_retention_decay.png"),
        }
        _write_metrics(summary, results_dir / "mixture_sweep_summary.json")
    return summary


def direction_transfer_analysis(
    horizon: int = 2, tail: int = 2, n_ctx: int = 8, seed: int = 0,
    n_fit: int = 6000, n_test: int = 6000, device=None,
    checkpoint_path=None, results_dir: Path = RESULTS_DIR, save: bool = True,
):
    """Is the spent coin carried in the *same* residual direction across positions?

    Fit a plain logistic probe per position (each position's coin-direction), then
    transfer the reveal-position probe to every position. High transfer accuracy and
    high cosine similarity in epoch >= 2 mean one persistent representation is carried
    forward, rather than the coin being re-encoded position by position.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = horizon + tail

    ckpt = Path(checkpoint_path) if checkpoint_path else (
        REPO_ROOT / "checkpoints" / f"mixture_h{horizon}_t{tail}.pt")
    if ckpt.exists():
        model, _ = load_checkpoint(ckpt, device)
    else:
        arch = ArchConfig.paper(process, n_ctx=n_ctx, seed=seed)
        cfg = TrainConfig.fast(seq_len=n_ctx, n_steps=8000, seed=seed)
        model = train(process, arch, cfg, verbose=False,
                      init_states_fn=process.aligned_init_states).model

    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
    Xf, Xt = _residuals_3d(model, fit.tokens), _residuals_3d(model, test.tokens)
    yf, yt = fit.z_first, test.z_first
    L = Xf.shape[1]

    clfs, perpos_acc, dirs = [], np.empty(L), np.empty((L, Xf.shape[2]))
    for p in range(L):
        clf = LogisticRegression(max_iter=5000).fit(Xf[:, p, :], yf)
        clfs.append(clf)
        perpos_acc[p] = accuracy_score(yt, clf.predict(Xt[:, p, :]))
        dirs[p] = clf.coef_.ravel()
    reveal = horizon
    transfer_acc = np.array([accuracy_score(yt, clfs[reveal].predict(Xt[:, p, :]))
                             for p in range(L)])
    dn = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    cos_to_reveal = (dn @ dn[reveal]).tolist()
    ret = slice(epoch_len, L)

    metrics = {
        "process": f"mixture(h={horizon},tail={tail})", "n_ctx": n_ctx,
        "reveal_position": reveal,
        "perpos_accuracy": [round(float(a), 4) for a in perpos_acc],
        "transfer_accuracy": [round(float(a), 4) for a in transfer_acc],
        "cosine_to_reveal_direction": [round(float(c), 4) for c in cos_to_reveal],
        "mean_transfer_acc_retention": float(np.mean(transfer_acc[ret])),
        "mean_cosine_retention": float(np.mean(np.array(cos_to_reveal)[ret])),
        "interpretation": (
            "Reveal-position probe transferred to epoch>=2 positions: high accuracy and "
            "high cosine similarity mean the coin is carried in the same residual direction "
            "(one persistent representation), not re-encoded per position."
        ),
    }
    fig = viz.plot_direction_transfer(
        np.arange(L), perpos_acc, transfer_acc, horizon, epoch_len,
        save_path=(results_dir / "mixture_direction_transfer.png") if save else None)
    if save:
        metrics["figure"] = _rel(results_dir / "mixture_direction_transfer.png")
        _write_metrics(metrics, results_dir / "metrics_mixture_direction.json")
    return {"metrics": metrics, "figure": fig}


def coin_position_geometry(horizon=2, tail=2, n_ctx=8, seed=0, n_fit=6000, n_test=6000,
                           device=None, checkpoint_path=None, results_dir: Path = RESULTS_DIR,
                           save: bool = True):
    """Full position x position transfer + cosine for the coin direction.

    Resolves the open question from the reveal-only check: do epoch >= 2 positions
    share ONE retained direction (bright off-diagonal block) or re-encode the coin
    per position (dark off-diagonal)?
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = horizon + tail
    ckpt = Path(checkpoint_path) if checkpoint_path else (
        REPO_ROOT / "checkpoints" / f"mixture_h{horizon}_t{tail}.pt")
    model, _ = load_checkpoint(ckpt, device)

    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
    Xf, Xt = _residuals_3d(model, fit.tokens), _residuals_3d(model, test.tokens)
    yf, yt = fit.z_first, test.z_first
    L = Xf.shape[1]

    clfs, dirs = [], np.empty((L, Xf.shape[2]))
    for p in range(L):
        clf = LogisticRegression(max_iter=5000).fit(Xf[:, p, :], yf)
        clfs.append(clf)
        dirs[p] = clf.coef_.ravel()
    transfer = np.array([[accuracy_score(yt, clfs[p].predict(Xt[:, q, :])) for q in range(L)]
                         for p in range(L)])
    dn = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    cosine = dn @ dn.T

    ret = list(range(epoch_len, L))
    off = lambda M: (float(np.mean([M[p, q] for p in ret for q in ret if p != q]))
                     if len(ret) > 1 else float("nan"))
    metrics = {
        "process": f"mixture(h={horizon},tail={tail})", "n_ctx": n_ctx, "epoch_len": epoch_len,
        "transfer_matrix": transfer.round(4).tolist(),
        "cosine_matrix": cosine.round(4).tolist(),
        "mean_within_retention_transfer_offdiag": off(transfer),
        "mean_within_retention_cosine_offdiag": off(cosine),
        "interpretation": ("High within-retention off-diagonal transfer/cosine => epoch>=2 "
                           "positions share one retained direction; low => coin is re-encoded "
                           "per position."),
    }
    fig = viz.plot_transfer_cosine_matrices(
        transfer, cosine, epoch_len,
        save_path=(results_dir / "mixture_position_geometry.png") if save else None,
        title=f"Coin direction across positions (mixture h={horizon})")
    if save:
        metrics["figure"] = _rel(results_dir / "mixture_position_geometry.png")
        _write_metrics(metrics, results_dir / "metrics_mixture_position_geometry.json")
    return {"metrics": metrics, "figure": fig}


def coin_depth_geometry(horizon=2, tail=2, n_ctx=8, seed=0, position=None,
                        n_fit=6000, n_test=6000, device=None, checkpoint_path=None,
                        results_dir: Path = RESULTS_DIR, save: bool = True):
    """Per-layer (resid_post) decode + cross-layer cosine at a fixed retention position.

    Depth analogue of the refusal finding: is the coin in a consistent direction across
    layers (high off-diagonal cosine) or depth-specific? Note resid_post is cumulative,
    so decodability is expected to be ~monotone in depth; the cosine is the informative part.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = horizon + tail
    position = epoch_len if position is None else position  # first retention position
    ckpt = Path(checkpoint_path) if checkpoint_path else (
        REPO_ROOT / "checkpoints" / f"mixture_h{horizon}_t{tail}.pt")
    model, _ = load_checkpoint(ckpt, device)
    nL = model.cfg.n_layers

    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
    cf = cache_residuals(model, fit.tokens, "resid_post")
    ct = cache_residuals(model, test.tokens, "resid_post")
    Nf, Nt = fit.tokens.shape[0], test.tokens.shape[0]
    yf, yt = fit.z_first, test.z_first

    dirs, acc = [], []
    for l in range(nL):
        Xf = cf[l].reshape(Nf, n_ctx, -1)[:, position, :]
        Xt = ct[l].reshape(Nt, n_ctx, -1)[:, position, :]
        clf = LogisticRegression(max_iter=5000).fit(Xf, yf)
        dirs.append(clf.coef_.ravel())
        acc.append(accuracy_score(yt, clf.predict(Xt)))
    dirs = np.array(dirs)
    dn = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    cos = dn @ dn.T

    metrics = {
        "process": f"mixture(h={horizon},tail={tail})", "position": position, "n_layers": nL,
        "layer_accuracy": [round(float(a), 4) for a in acc],
        "layer_cosine_matrix": cos.round(4).tolist(),
        "mean_offdiag_cosine": float(np.mean([cos[i, j] for i in range(nL)
                                              for j in range(nL) if i != j])),
        "interpretation": ("Per-layer decode of the coin at a retention position and cross-layer "
                           "cosine. High off-diagonal cosine => consistent direction across depth "
                           "(refusal-like). resid_post is cumulative, so accuracy is ~monotone in depth."),
    }
    fig = viz.plot_depth_consistency(
        list(range(nL)), acc, cos, position,
        save_path=(results_dir / "mixture_depth_geometry.png") if save else None,
        title=f"Coin across depth at position {position} (mixture h={horizon})")
    if save:
        metrics["figure"] = _rel(results_dir / "mixture_depth_geometry.png")
        _write_metrics(metrics, results_dir / "metrics_mixture_depth_geometry.json")
    return {"metrics": metrics, "figure": fig}


def causal_ablation(horizon=2, tail=2, n_ctx=8, seed=0, n_fit=6000, n_test=6000,
                    device=None, checkpoint_path=None, results_dir: Path = RESULTS_DIR,
                    save: bool = True):
    """Is the *retained* coin causally used, or inert?

    Directional ablation: at chosen positions, project the coin's diff-in-means
    direction out of every layer's resid_post, then measure per-position next-token
    loss. Three conditions:
      * clean,
      * **ablate retained** (epoch >= 2 positions) — expect loss ~ unchanged if inert,
      * **ablate used** (the epoch-1 reveal position, which must carry the coin to
        predict the next indicator) — positive control; expect a loss spike.
    A decodability-drop check confirms the ablation actually removed the coin.
    """
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = horizon + tail
    ckpt = Path(checkpoint_path) if checkpoint_path else (
        REPO_ROOT / "checkpoints" / f"mixture_h{horizon}_t{tail}.pt")
    model, _ = load_checkpoint(ckpt, device)
    nL, d_model, L = model.cfg.n_layers, model.cfg.d_model, n_ctx

    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)

    # diff-in-means coin direction per (layer, position) from the fit set
    cf = cache_residuals(model, fit.tokens, "resid_post")
    yf, Nf = fit.z_first, fit.tokens.shape[0]
    dirs = {}
    for l in range(nL):
        R = cf[l].reshape(Nf, L, d_model)
        for p in range(L):
            v = R[yf == 1, p].mean(0) - R[yf == 0, p].mean(0)
            n = np.linalg.norm(v)
            dirs[(l, p)] = torch.tensor(v / n if n > 0 else v, dtype=torch.float32, device=device)

    def hooks_for(positions):
        def mk(l):
            vs = {p: dirs[(l, p)] for p in positions}
            def hook(act, hook):
                for p, v in vs.items():
                    act[:, p, :] = act[:, p, :] - (act[:, p, :] @ v).unsqueeze(-1) * v
                return act
            return hook
        return [(f"blocks.{l}.hook_resid_post", mk(l)) for l in range(nL)]

    def per_pos_loss(hooks):
        with torch.no_grad():
            logits = model.run_with_hooks(test.tokens, fwd_hooks=hooks, return_type="logits")
            logp = F.log_softmax(logits[:, :-1, :], dim=-1)
            tgt = test.tokens[:, 1:]
            nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            return nll.mean(0).cpu().numpy()

    retain = list(range(epoch_len, L))
    use_pos = horizon  # epoch-1 reveal: must carry the coin to predict token horizon+1
    clean = per_pos_loss([])
    ablate_retained = per_pos_loss(hooks_for(retain))
    ablate_used = per_pos_loss(hooks_for([use_pos]))

    # sanity: did the retained-ablation actually remove the coin? decode z_first at the
    # first retention position from the (ablated) final-layer resid.
    captured = {}
    def capture(act, hook):
        captured["r"] = act.detach().cpu().numpy()
        return act
    with torch.no_grad():
        model.run_with_hooks(
            test.tokens,
            fwd_hooks=hooks_for(retain) + [(f"blocks.{nL-1}.hook_resid_post", capture)],
            return_type=None)
    abl_final = captured["r"].reshape(test.tokens.shape[0], L, d_model)
    clean_final_fit = cf[nL-1].reshape(Nf, L, d_model)
    clean_final_test = cache_residuals(model, test.tokens, "resid_post")[nL-1].reshape(
        test.tokens.shape[0], L, d_model)
    pos0 = retain[0]
    probe = LogisticRegression(max_iter=5000).fit(clean_final_fit[:, pos0, :], yf)
    acc_clean = accuracy_score(test.z_first, probe.predict(clean_final_test[:, pos0, :]))
    acc_ablated = accuracy_score(test.z_first, probe.predict(abl_final[:, pos0, :]))

    pred_pos = list(range(L - 1))
    metrics = {
        "process": f"mixture(h={horizon},tail={tail})", "n_ctx": n_ctx,
        "use_position": use_pos, "retention_positions": retain,
        "clean_loss_by_position": [round(float(x), 4) for x in clean],
        "ablate_retained_loss_by_position": [round(float(x), 4) for x in ablate_retained],
        "ablate_used_loss_by_position": [round(float(x), 4) for x in ablate_used],
        "total_loss_clean": round(float(clean.mean()), 4),
        "total_loss_ablate_retained": round(float(ablate_retained.mean()), 4),
        "total_loss_ablate_used": round(float(ablate_used.mean()), 4),
        "delta_at_use_position": round(float(ablate_used[use_pos] - clean[use_pos]), 4),
        "max_abs_delta_retention": round(float(np.max(np.abs(
            ablate_retained[retain[0]:] - clean[retain[0]:]))), 4),
        "z_first_decode_at_retention_clean": round(float(acc_clean), 4),
        "z_first_decode_at_retention_ablated": round(float(acc_ablated), 4),
        "interpretation": ("Ablating the retained coin (epoch>=2) leaves next-token loss "
                           "unchanged while it is fully removed (decodability -> chance), yet "
                           "ablating the same coin where it is USED (the reveal) spikes the loss. "
                           "=> the retained copy is decodable but causally inert: the model is an "
                           "optimal predictor carrying genuinely non-minimal information."),
    }
    fig = viz.plot_ablation_loss(
        pred_pos, clean, ablate_used, ablate_retained, use_pos, epoch_len,
        save_path=(results_dir / "mixture_ablation.png") if save else None,
        title=f"Causal ablation of the coin (mixture h={horizon})")
    if save:
        metrics["figure"] = _rel(results_dir / "mixture_ablation.png")
        _write_metrics(metrics, results_dir / "metrics_mixture_ablation.json")
    return {"metrics": metrics, "figure": fig}


def capacity_pressure_sweep(
    widths=(2, 3, 4, 6, 8, 16, 64),
    seeds=(0, 1, 2),
    horizon: int = 2,
    tail: int = 2,
    n_ctx: int = 8,
    n_steps: int = 10_000,
    n_fit: int = 4000,
    n_test: int = 4000,
    converged_tol: float = 0.02,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Does minimality emerge when residual bandwidth is scarce?

    With d_model=64 the model retains the spent coin — unsurprising: the stream has
    ~60 spare dimensions and the loss has no forgetting term. This sweep shrinks
    ONLY the residual stream (d_model, with d_head=min(8, d_model); the MLP stays
    at width 256 so compute stays generous and the squeeze is specifically on
    representational bandwidth) and asks where, if anywhere, the per-position state
    converges to the minimal belief and drops the spent coin.

    The essential control is the loss gap: a retention drop only counts as
    *emergent minimality* at widths where the model still reaches the optimal
    floor. (Failing to learn the task at all would also kill retention, for the
    boring reason.) Each (width, seed) record is written incrementally.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = horizon + tail
    floor = _mixture_floor_nats(process, n_ctx, seed=999)
    records = []
    json_path = results_dir / "mixture_capacity_sweep.json"

    for w in widths:
        for seed in seeds:
            try:
                arch = ArchConfig.paper(process, n_ctx=n_ctx, seed=seed,
                                        d_model=w, d_head=min(8, w))
                cfg = TrainConfig.fast(seq_len=n_ctx, n_steps=n_steps, seed=seed)
                result = train(process, arch, cfg, verbose=False,
                               init_states_fn=process.aligned_init_states,
                               process_name=f"mix-cap(w={w},s={seed})")
                model = result.model
                loss = _eval_aligned_loss(model, process, n_ctx, device)
                fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
                test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
                _, acc = probe_label_by_position(model, fit.tokens, fit.z_first,
                                                 test.tokens, test.z_first)
                # Needed-info control: the *current* (second-epoch) coin at the last
                # position, which any converged model must carry.
                Xf = _residuals_3d(model, fit.tokens)[:, -1, :]
                Xt = _residuals_3d(model, test.tokens)[:, -1, :]
                cur = LogisticRegression(max_iter=2000).fit(Xf, fit.z_labels[:, -1])
                cur_acc = accuracy_score(test.z_labels[:, -1], cur.predict(Xt))

                prefix, revealed, retention = acc[:horizon], acc[horizon:epoch_len], acc[epoch_len:]
                rec = {
                    "d_model": w, "seed": seed, "d_head": min(8, w),
                    "n_params": int(sum(p.numel() for p in model.parameters())),
                    "loss_nats": round(loss, 4), "floor_nats": round(floor, 4),
                    "gap_nats": round(loss - floor, 4),
                    "converged": bool(loss - floor < converged_tol),
                    "mean_prefix_acc": float(prefix.mean()),
                    "mean_revealed_acc": float(revealed.mean()),
                    "mean_retention_acc": float(retention.mean()),
                    "current_coin_acc_last": float(cur_acc),
                    "accuracy_by_position": [round(float(a), 4) for a in acc],
                }
                records.append(rec)
                print(f"[cap] w={w:>2} seed={seed} gap={loss-floor:+.4f} "
                      f"conv={rec['converged']} retention={rec['mean_retention_acc']:.3f} "
                      f"current={cur_acc:.3f}", flush=True)
                del model, result, fit, test
            except Exception as e:
                records.append({"d_model": w, "seed": seed, "error": repr(e)})
                print(f"[cap] w={w} seed={seed} FAILED: {e!r}", flush=True)
            if save:
                _write_metrics({"floor_nats": floor, "records": records}, json_path)

    ok = [r for r in records if "error" not in r]
    by_w = {}
    for w in widths:
        rs_all = [r for r in ok if r["d_model"] == w]
        if not rs_all:
            continue
        conv = [r for r in rs_all if r["converged"]]
        rs = conv or rs_all  # stats from converged runs when any exist
        ret = np.array([r["mean_retention_acc"] for r in rs])
        gaps = np.array([r["gap_nats"] for r in rs_all])
        by_w[w] = {
            "retention_mean": float(ret.mean()), "retention_std": float(ret.std()),
            "revealed": float(np.mean([r["mean_revealed_acc"] for r in rs])),
            "prefix": float(np.mean([r["mean_prefix_acc"] for r in rs])),
            "current_coin": float(np.mean([r["current_coin_acc_last"] for r in rs])),
            "gap_mean": float(gaps.mean()), "gap_std": float(gaps.std()),
            "n_converged": len(conv), "n_runs": len(rs_all),
        }

    summary = {
        "config": {"widths": list(widths), "seeds": list(seeds), "horizon": horizon,
                   "tail": tail, "n_ctx": n_ctx, "n_steps": n_steps,
                   "converged_tol": converged_tol, "d_mlp": "256 (fixed, deliberately)"},
        "floor_nats": floor,
        "by_width": {str(w): by_w[w] for w in by_w},
        "records": records,
    }
    if save and by_w:
        viz.plot_capacity_pressure(
            by_w, converged_tol, len(seeds),
            save_path=results_dir / "mixture_capacity_pressure.png")
        summary["figure"] = _rel(results_dir / "mixture_capacity_pressure.png")
        _write_metrics(summary, results_dir / "mixture_capacity_summary.json")
    return summary


# ====================================================================== #
# Retention ledger: does the model hold ALL the spent coins at once?
# ====================================================================== #
def retention_ledger(
    horizon: int = 2,
    tail: int = 2,
    n_epochs: int = 6,
    d_model: int = 64,
    n_steps: int = 10_000,
    n_fit: int = 4000,
    n_test: int = 4000,
    seed: int = 0,
    converged_tol: float = 0.02,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Many-epoch retention: decode EVERY epoch's coin at EVERY position.

    Phase 2 established that one spent coin is retained across one epoch boundary.
    This extends the context to ``n_epochs`` epochs (``n_ctx = n_epochs *
    (horizon + tail)``), so by the end the stream holds several coins of different
    ages — and asks whether the model keeps a *ledger* of all of them.

    The accuracy matrix ``acc[e, p]`` (epoch-e's coin decoded at position p) has
    built-in sanity structure: ~0.5 before epoch e begins (that coin hasn't been
    flipped yet — decoding it would mean predicting a future fair coin) and ~0.5
    through epoch e's prefix (information-theoretically absent, the E4 control).
    The science is the region after each reveal: a full bright band to the right
    edge means every dead coin is retained simultaneously; sagging bands for old
    coins mean graceful forgetting with age. Run at small ``d_model`` to ask
    whether *load* (several dead coins + the live task) forces the oldest out.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    device = get_device(device)
    process = MixtureProcess(horizon, tail)
    epoch_len = process.epoch_len
    n_ctx = n_epochs * epoch_len

    arch = ArchConfig.paper(process, n_ctx=n_ctx, seed=seed,
                            d_model=d_model, d_head=min(8, d_model))
    cfg = TrainConfig.fast(seq_len=n_ctx, n_steps=n_steps, seed=seed)
    result = train(process, arch, cfg, verbose=False,
                   init_states_fn=process.aligned_init_states,
                   process_name=f"mix-ledger(w={d_model})")
    model = result.model
    floor = _mixture_floor_nats(process, n_ctx, seed=seed + 999)
    loss = _eval_aligned_loss(model, process, n_ctx, device)
    print(f"[ledger w={d_model}] trained: loss={loss:.4f} floor={floor:.4f} "
          f"gap={loss - floor:+.4f}", flush=True)
    if save:
        save_checkpoint(result, REPO_ROOT / "checkpoints" /
                        f"mixture_ledger_w{d_model}.pt")

    fit = make_mixture_eval_set(process, n_fit, n_ctx, seed + 1, device)
    test = make_mixture_eval_set(process, n_test, n_ctx, seed + 2, device)
    Xf, Xt = _residuals_3d(model, fit.tokens), _residuals_3d(model, test.tokens)

    acc = np.zeros((n_epochs, n_ctx))
    for e in range(n_epochs):
        yf = fit.z_labels[:, e * epoch_len]
        yt = test.z_labels[:, e * epoch_len]
        for p in range(n_ctx):
            clf = LogisticRegression(max_iter=2000).fit(Xf[:, p, :], yf)
            acc[e, p] = accuracy_score(yt, clf.predict(Xt[:, p, :]))
        print(f"[ledger w={d_model}] coin {e + 1}/{n_epochs} probed "
              f"(final-pos acc {acc[e, -1]:.3f})", flush=True)

    reveals = [e * epoch_len + horizon for e in range(n_epochs)]
    # retention-by-age: mean accuracy at a given offset since reveal, over coins
    by_age = {}
    for off in range(n_ctx - horizon):
        vals = [acc[e, reveals[e] + off] for e in range(n_epochs)
                if reveals[e] + off < n_ctx]
        if vals:
            by_age[off] = round(float(np.mean(vals)), 4)
    future_coin = float(np.mean([acc[e, :e * epoch_len].mean()
                                 for e in range(1, n_epochs)]))
    prefix_ctrl = float(np.mean([acc[e, e * epoch_len:reveals[e]].mean()
                                 for e in range(n_epochs)]))

    metrics = {
        "process": f"mixture(h={horizon},tail={tail})",
        "d_model": d_model, "n_epochs": n_epochs, "n_ctx": n_ctx, "seed": seed,
        "training": {"final_loss_nats": round(loss, 4),
                     "floor_nats": round(floor, 4),
                     "gap_nats": round(loss - floor, 4),
                     "converged": bool(loss - floor < converged_tol)},
        "accuracy_matrix": acc.round(4).tolist(),
        "ledger_at_final_position": [round(float(acc[e, -1]), 4)
                                     for e in range(n_epochs)],
        "retention_by_age": by_age,
        "future_coin_mean_acc": round(future_coin, 4),
        "prefix_control_mean_acc": round(prefix_ctrl, 4),
        "interpretation": (
            "acc[e, p] = decoding epoch-e's coin at position p from the all-layer "
            "residual. ~0.5 before epoch e (future coin) and through its prefix "
            "(E4 control) validate the probe; the post-reveal band is the ledger: "
            "flat ~1.0 to the right edge = all dead coins retained simultaneously; "
            "sagging old-coin bands = forgetting with age."),
    }
    fig = viz.plot_retention_ledger(
        acc, horizon, epoch_len,
        save_path=(results_dir / f"retention_ledger_w{d_model}.png") if save else None,
        title=f"Retention ledger — {n_epochs} epochs, d_model={d_model} "
              f"(gap {loss - floor:+.3f} nats)")
    if save:
        metrics["figure"] = _rel(results_dir / f"retention_ledger_w{d_model}.png")
        _write_metrics(metrics, results_dir / f"metrics_retention_ledger_w{d_model}.json")
    return {"metrics": metrics, "acc": acc, "figure": fig, "model": model}


# ====================================================================== #
# Norm vs uncertainty: does the residual's LENGTH carry belief confidence?
# ====================================================================== #
def norm_confidence_analysis(
    checkpoint_path,
    process,
    n_seqs: int = 4000,
    seed: int = 0,
    hook: str = "resid_post",
    min_cell: int = 50,
    device=None,
):
    """Within-position correlation between residual L2 norm and belief uncertainty.

    Phase 1/2 study the residual's *direction* (the belief geometry). This asks an
    orthogonal question: does its *norm* track the optimal observer's uncertainty —
    the Shannon entropy of the analytic belief (and, separately, of the optimal
    next-token distribution)?

    Methodological notes (the design IS the result here):

    * **Position is a confound.** Norms grow systematically with position (learned
      positional embeddings, accumulated writes) and belief entropy falls with
      position (synchronization). A pooled correlation would mostly measure
      position. All correlations are therefore computed *within* a (layer,
      position) cell, across sequences, then aggregated.
    * **Token identity is a sub-confound.** Within a position, the current token's
      embedding contributes to the norm and also moves the belief. A stricter
      within-(position, current-token) Spearman is reported alongside.
    * **Degenerate cells are skipped.** At position 0 (and any symmetric cell) all
      histories give equal-entropy beliefs (process symmetry), so the correlation
      is undefined there; cells with ~zero entropy variance are excluded and
      counted.
    * **The mixture process is excluded by design**: epoch-aligned phase determines
      its belief entropy exactly, so within-position entropy variance is ~0 at
      every position — there is nothing to correlate. Mess3 (continuous entropy on
      the fractal) and RRXOR (discrete entropy levels across histories) are the
      right testbeds.
    * **LayerNorm null.** Each block reads the stream through LayerNorm, which
      discards scale; the network has no first-order reason to *use* the norm. A
      near-zero correlation is therefore a meaningful null ("confidence is purely
      directional"), not a failed experiment.
    """
    from scipy.stats import spearmanr

    device = get_device(device)
    model, ckpt = load_checkpoint(checkpoint_path, device)
    seq_len = ckpt["arch"]["n_ctx"]
    data = make_eval_set(process, n_seqs, seq_len, seed + 11, device)

    Hb = B.entropy(data.beliefs, axis=-1, units="bits")            # [N, L] belief entropy
    nd = np.einsum("nli,xij->nlx", data.beliefs, process.T)        # P(next | belief)
    Hx = B.entropy(nd, axis=-1, units="bits")                      # [N, L] next-token entropy

    N, L = data.tokens.shape
    usable = [t for t in range(L) if Hb[:, t].std() > 1e-9]
    skipped = [t for t in range(L) if t not in usable]

    cached = cache_residuals(model, data.tokens, hook)
    per_layer = {}
    best = None
    for l in sorted(cached):
        R = cached[l].reshape(N, L, -1)
        norms = np.linalg.norm(R, axis=-1)                         # [N, L]
        rb, rx, rbt = [], [], []
        for t in usable:
            rho_b = float(spearmanr(norms[:, t], Hb[:, t]).statistic)
            rho_x = float(spearmanr(norms[:, t], Hx[:, t]).statistic)
            rb.append(rho_b)
            rx.append(rho_x)
            # stricter control: within (position, current token)
            cell = []
            for tok in range(process.n_symbols):
                m = data.emissions[:, t] == tok
                if m.sum() >= min_cell and Hb[m, t].std() > 1e-9:
                    cell.append(float(spearmanr(norms[m, t], Hb[m, t]).statistic))
            if cell:
                rbt.append(float(np.mean(cell)))
            if best is None or abs(rho_b) > abs(best["rho"]):
                best = {"layer": int(l), "position": int(t), "rho": rho_b,
                        "norm": norms[:, t].copy(), "H": Hb[:, t].copy()}
        per_layer[l] = {
            "rho_belief_mean": float(np.mean(rb)), "rho_belief_std": float(np.std(rb)),
            "rho_nexttok_mean": float(np.mean(rx)), "rho_nexttok_std": float(np.std(rx)),
            "rho_belief_per_position": [round(v, 4) for v in rb],
            "rho_nexttok_per_position": [round(v, 4) for v in rx],
            "rho_belief_within_pos_and_token_mean":
                float(np.mean(rbt)) if rbt else None,
        }

    # how entangled the two uncertainty measures are (context for reading the bars)
    corr_HbHx = float(spearmanr(Hb[:, usable].ravel(), Hx[:, usable].ravel()).statistic)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "n_seqs": n_seqs, "seq_len": seq_len, "hook": hook,
        "usable_positions": usable, "skipped_positions_degenerate_entropy": skipped,
        "per_layer": {str(l): per_layer[l] for l in per_layer},
        "strongest_cell": {"layer": best["layer"], "position": best["position"],
                           "rho": round(best["rho"], 4)},
        "spearman_Hbelief_vs_Hnexttok": round(corr_HbHx, 4),
    }
    return metrics, best, per_layer


def run_norm_confidence(
    n_seqs: int = 4000,
    seed: int = 0,
    device=None,
    results_dir: Path = RESULTS_DIR,
    save: bool = True,
):
    """Run the norm-vs-uncertainty analysis on the Mess3 and RRXOR checkpoints."""
    out = {}
    ckpt_dir = REPO_ROOT / "checkpoints"
    for name, proc, ckpt in (("mess3", Mess3(), ckpt_dir / "mess3_fast.pt"),
                             ("rrxor", RRXOR(), ckpt_dir / "rrxor_fast.pt")):
        metrics, best, per_layer = norm_confidence_analysis(
            ckpt, proc, n_seqs=n_seqs, seed=seed, device=device)
        fig = viz.plot_norm_confidence(
            best, per_layer,
            save_path=(results_dir / f"norm_confidence_{name}.png") if save else None,
            title=f"{name}: residual norm vs belief uncertainty (within-position)")
        metrics["figure"] = _rel(results_dir / f"norm_confidence_{name}.png")
        out[name] = {"metrics": metrics, "figure": fig}
    if save:
        _write_metrics({k: v["metrics"] for k, v in out.items()},
                       results_dir / "metrics_norm_confidence.json")
    return out
