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
from data import make_eval_set, get_device, sample_tokens
from hmms import Mess3, RRXOR
from probe import (
    ProbeResult,
    cache_residuals,
    flatten_beliefs,
    probe_all_layers,
    probe_concat_layers,
    best_layer,
)
from train import load_checkpoint
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
