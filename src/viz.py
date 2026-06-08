"""Figures: the predicted belief geometry vs. what we recover from the model.

The headline figure (:func:`plot_simplex_comparison`) puts the analytic Mess3
belief fractal next to the belief states linearly decoded from the residual
stream, coloured identically. If the paper's claim holds they look the same.

Other figures: training loss vs the analytic floor, and (for RRXOR) per-layer
R^2 showing the representation is distributed across layers.

Plot functions create and return a Matplotlib figure and optionally save it;
they don't set a global backend, so they work both in notebooks (inline) and in
headless scripts (call with ``save_path=...``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import beliefs as B
from probe import ProbeResult, simplex_points


# --------------------------------------------------------------------- #
# Simplex helpers
# --------------------------------------------------------------------- #
def _draw_triangle(ax, labels=("A", "B", "C"), dark=False):
    """Draw the 2-simplex outline with corner labels and clean axes."""
    line_c = "0.8" if dark else "k"
    text_c = "0.7" if dark else "0.3"
    if dark:
        ax.set_facecolor("black")
    v = B.simplex_triangle_vertices()
    tri = np.vstack([v, v[0]])
    ax.plot(tri[:, 0], tri[:, 1], color=line_c, lw=1.0, zorder=3)
    offsets = [(-0.03, -0.04), (0.03, -0.04), (0.0, 0.04)]
    ha = ["right", "left", "center"]
    va = ["top", "top", "bottom"]
    for i, (lab, off) in enumerate(zip(labels, offsets)):
        ax.annotate(f"state {i} ({lab})", v[i], xytext=(v[i][0] + off[0], v[i][1] + off[1]),
                    ha=ha[i], va=va[i], fontsize=9, color=text_c)
    ax.set_aspect("equal")
    ax.axis("off")
    m = 0.12
    ax.set_xlim(-m, 1 + m)
    ax.set_ylim(-m, np.sqrt(3) / 2 + m)


def _scatter_simplex(ax, xy, rgb, s=2.0, alpha=0.35, max_points=120_000, seed=0):
    if xy.shape[0] > max_points:
        idx = np.random.default_rng(seed).choice(xy.shape[0], max_points, replace=False)
        xy, rgb = xy[idx], rgb[idx]
    ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(rgb, 0, 1), s=s, alpha=alpha,
               linewidths=0, rasterized=True, zorder=2)


# --------------------------------------------------------------------- #
# Headline figure: predicted fractal vs recovered cloud (Mess3)
# --------------------------------------------------------------------- #
def plot_simplex_comparison(
    result: ProbeResult,
    symbol_labels=("A", "B", "C"),
    title: str | None = None,
    save_path: str | Path | None = None,
    s: float = 1.1,
    alpha: float = 0.25,
    max_points: int = 400_000,
    dark: bool = True,
):
    """Side-by-side: analytic belief geometry vs. belief decoded from residuals.

    Both clouds are coloured by the *true* belief (RGB == belief), so the
    comparison is like-for-like; a visual match is the headline result.
    """
    true_xy, pred_xy, rgb = simplex_points(result)
    title_c = "0.92" if dark else "k"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    if dark:
        fig.patch.set_facecolor("black")
    _draw_triangle(axes[0], symbol_labels, dark=dark)
    _scatter_simplex(axes[0], true_xy, rgb, s, alpha, max_points)
    axes[0].set_title("Predicted belief geometry\n(analytic optimal observer)",
                      fontsize=11, color=title_c)

    _draw_triangle(axes[1], symbol_labels, dark=dark)
    _scatter_simplex(axes[1], pred_xy, rgb, s, alpha, max_points)
    axes[1].set_title(
        f"Recovered from residual stream\n(linear probe, layer {result.layer}, "
        f"R²={result.r2:.3f})", fontsize=11, color=title_c)

    if title:
        fig.suptitle(title, fontsize=13, y=1.0, color=title_c)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path, dark=dark)
    return fig


def plot_simplex_single(xy, rgb, title="", labels=("A", "B", "C"),
                        save_path=None, s=2.0, alpha=0.35, max_points=120_000):
    """Plot a single belief cloud in the simplex (e.g. the theory cloud alone)."""
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    _draw_triangle(ax, labels)
    _scatter_simplex(ax, xy, rgb, s, alpha, max_points)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# --------------------------------------------------------------------- #
# Training curve
# --------------------------------------------------------------------- #
def plot_training_curve(
    history: dict,
    floor_nats: float,
    entropy_rate_bits: float | None = None,
    title: str = "Training loss vs analytic floor",
    save_path: str | Path | None = None,
):
    """Plot CE loss against the optimal in-context loss (and entropy rate)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["step"], history["loss"], label="train loss (nats)", color="C0", lw=1.5)
    ax.axhline(floor_nats, color="k", ls="--", lw=1.2,
               label=f"optimal in-context loss = {floor_nats:.3f} nats")
    if entropy_rate_bits is not None:
        ax.axhline(entropy_rate_bits * np.log(2), color="0.6", ls=":", lw=1.2,
                   label=f"entropy rate = {entropy_rate_bits:.3f} bits")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy (nats)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


# --------------------------------------------------------------------- #
# Per-layer R^2 (RRXOR distributed representation)
# --------------------------------------------------------------------- #
def plot_layer_r2(
    per_layer: dict[int, ProbeResult],
    concat: ProbeResult | None = None,
    title: str = "Belief recovery R² by layer",
    save_path: str | Path | None = None,
):
    """Bar chart of probe R^2 at each layer (+ the across-layer concatenation)."""
    layers = sorted(per_layer)
    vals = [per_layer[l].r2 for l in layers]
    labels = [f"layer {l}" for l in layers]
    colors = ["C0"] * len(layers)
    if concat is not None:
        labels.append("concat\n(all layers)")
        vals.append(concat.r2)
        colors.append("C3")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("probe R² (held-out)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_pred_vs_true(result: ProbeResult, state_labels=None,
                      title=None, save_path=None, max_points=20_000):
    """Predicted vs. true belief per coordinate -- a dimension-agnostic R^2 view
    (used for RRXOR, whose 5-state belief can't be drawn in a 2-simplex)."""
    S = result.y_true.shape[1]
    n = result.y_true.shape[0]
    idx = (np.random.default_rng(0).choice(n, max_points, replace=False)
           if n > max_points else slice(None))
    fig, axes = plt.subplots(1, S, figsize=(2.6 * S, 2.8), squeeze=False)
    axes = axes[0]
    for k in range(S):
        ax = axes[k]
        ax.scatter(result.y_true[idx, k], result.y_pred[idx, k], s=3, alpha=0.2,
                   color="C0", rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        lab = state_labels[k] if state_labels else f"state {k}"
        ax.set_title(f"{lab}\nR²={result.r2_per_coord[k]:.3f}", fontsize=9)
        ax.set_xlabel("true"); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        if k == 0:
            ax.set_ylabel("predicted")
        ax.set_aspect("equal")
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_future_information(fi: dict, title=None, save_path=None, max_points=80_000):
    """Two-panel: (left) belief vs next-token-distribution pairwise distances;
    (right) R² of recovering belief from the residual vs from the next-token dist.

    The left panel's mass at next-token distance ~0 with belief distance > 0 is the
    visual of "the belief distinguishes positions the next token cannot". The right
    panel quantifies it: the residual recovers the belief; the next-token
    distribution (a lossy function of belief) cannot.
    """
    d_b, d_p = fi["d_belief"], fi["d_nexttok"]
    if d_b.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(d_b.shape[0], max_points, replace=False)
        d_b, d_p = d_b[idx], d_p[idx]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    hb = axes[0].hexbin(d_p, d_b, gridsize=45, cmap="viridis", bins="log", mincnt=1)
    fig.colorbar(hb, ax=axes[0], label="pair count (log)")
    axes[0].set_xlabel("‖Δ next-token distribution‖")
    axes[0].set_ylabel("‖Δ belief state‖")
    axes[0].set_title("Pairs equal in next-token,\ndiffering in belief (mass at left edge)",
                      fontsize=10)

    r2r = fi["r2_residual_to_belief"]
    r2p = fi["r2_nexttoken_to_belief"]
    bars = axes[1].bar(["residual\n→ belief", "next-token dist\n→ belief"],
                       [r2r, r2p], color=["C3", "0.6"])
    for b, v in zip(bars, [r2r, r2p]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=10)
    axes[1].set_ylabel("R² (recover belief)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Belief is in the residual stream,\nnot derivable from next-token alone",
                      fontsize=10)
    axes[1].grid(axis="y", alpha=0.25)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_generator_decodability(
    positions, model_acc, upper_acc, horizon, epoch_len,
    title=None, save_path=None,
):
    """Accuracy of decoding the *first* epoch's generator label vs token position.

    The first epoch's Z is revealed at its first indicator (position ``horizon``)
    and remains in the raw history afterwards (upper-bound curve). In later epochs
    it is neither in the current input nor predictively needed, so the model's
    accuracy there measures **retention**: tracking the upper bound = the model
    keeps predictively-useless identity; dropping to chance = it discards it.
    """
    positions = np.asarray(positions)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    n_ep = (len(positions) + epoch_len - 1) // epoch_len
    if n_ep > 1:  # shade everything after the first epoch (the retention region)
        ax.axvspan(epoch_len - 0.5, len(positions) - 0.5, color="C0", alpha=0.07,
                   label="retention region (epoch ≥ 2)")
    for e in range(1, n_ep):
        ax.axvline(e * epoch_len - 0.5, color="0.85", lw=1)
    ax.axhline(0.5, color="0.6", ls=":", lw=1, label="chance")
    ax.plot(positions, upper_acc, color="k", ls="--", lw=1.4, marker="s", ms=4,
            label="upper bound (label is in history)")
    ax.plot(positions, model_acc, color="C3", lw=2, marker="o", ms=5,
            label="model residual (linear probe)")
    ax.axvline(horizon, color="C2", lw=1.2, ls="-.", alpha=0.7,
               label=f"reveal (position {horizon})")
    ax.set_xlabel("token position")
    ax.set_ylabel("decode first-epoch generator (accuracy)")
    ax.set_ylim(0.4, 1.03)
    ax.set_xticks(list(positions))
    ax.set_title(title or "Mixture: where is the generator label represented?")
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_retention_vs_horizon(by_i, save_path=None, title=None):
    """Mean retention accuracy (decode spent coin in epoch >= 2) vs divergence horizon,
    with seed error bars; reference curves for the post-reveal and prefix regimes."""
    iv = sorted(by_i)
    means = [by_i[i]["mean"] for i in iv]
    stds = [by_i[i]["std"] for i in iv]
    rev = [by_i[i]["revealed"] for i in iv]
    pre = [by_i[i]["prefix"] for i in iv]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0.5, color="0.6", ls=":", lw=1, label="chance")
    ax.plot(iv, rev, marker="s", color="0.45", ls="--", lw=1.2, label="epoch 1, after reveal")
    ax.errorbar(iv, means, yerr=stds, marker="o", color="C3", capsize=3, lw=2,
                label="retention (epoch ≥ 2)")
    ax.plot(iv, pre, marker="^", color="0.7", ls=":", lw=1, label="prefix (control)")
    ax.set_xscale("log")
    ax.set_xticks(iv)
    ax.set_xticklabels([str(i) for i in iv])
    ax.set_xlabel("divergence horizon  i")
    ax.set_ylabel("decode first-epoch coin (accuracy)")
    ax.set_ylim(0.4, 1.03)
    ax.set_title(title or "Retention of the spent coin vs. horizon (± seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_retention_decay(decay_by_i, save_path=None, title=None):
    """Decode accuracy vs. tokens-since-reveal, one curve per horizon (seed-averaged).
    Flat near the upper bound = no decay (persistent memory); a downward slope = decay."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0.5, color="0.6", ls=":", lw=1, label="chance")
    for i in sorted(decay_by_i):
        x, acc = decay_by_i[i]
        ax.plot(x, acc, marker=".", lw=1.5, label=f"i={i}")
    ax.axvline(0, color="C2", ls="-.", lw=1, alpha=0.6)
    ax.set_xlabel("tokens since the first-epoch coin was revealed")
    ax.set_ylabel("decode coin (accuracy)")
    ax.set_ylim(0.4, 1.03)
    ax.set_title(title or "Does the retained coin decay with distance?")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_direction_transfer(positions, perpos_acc, transfer_acc, horizon, epoch_len,
                            save_path=None, title=None):
    """Per-position probe vs. the reveal-position probe transferred to other positions.
    If the transferred probe keeps working in epoch >= 2, the coin is carried in the
    *same* direction (one persistent representation), not re-encoded position by position."""
    positions = np.asarray(positions)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    n_ep = (len(positions) + epoch_len - 1) // epoch_len
    if n_ep > 1:
        ax.axvspan(epoch_len - 0.5, len(positions) - 0.5, color="C0", alpha=0.07,
                   label="retention region (epoch ≥ 2)")
    ax.axhline(0.5, color="0.6", ls=":", lw=1, label="chance")
    ax.plot(positions, perpos_acc, marker="o", color="C3", lw=2,
            label="probe trained per position")
    ax.plot(positions, transfer_acc, marker="s", color="C0", lw=1.6, ls="--",
            label="reveal-position probe, transferred")
    ax.axvline(horizon, color="C2", ls="-.", lw=1, alpha=0.7, label=f"reveal (pos {horizon})")
    ax.set_xlabel("token position")
    ax.set_ylabel("decode first-epoch coin (accuracy)")
    ax.set_ylim(0.4, 1.03)
    ax.set_xticks(list(positions))
    ax.set_title(title or "Is the coin carried in the same direction?")
    ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_transfer_cosine_matrices(transfer, cosine, epoch_len, save_path=None, title=None):
    """Heatmaps of position->position probe-transfer accuracy and coin-direction cosine.
    A bright off-diagonal block among epoch>=2 positions = one shared retained
    direction; a dark off-diagonal = the coin is re-encoded per position."""
    transfer, cosine = np.asarray(transfer), np.asarray(cosine)
    L = transfer.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, M, name, vmin in ((axes[0], transfer, "transfer accuracy", 0.5),
                              (axes[1], cosine, "direction cosine", -1.0)):
        im = ax.imshow(M, origin="lower", cmap="viridis", vmin=vmin, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("evaluated at position")
        ax.set_ylabel("probe trained at position")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(L)); ax.set_yticks(range(L))
        ax.axvline(epoch_len - 0.5, color="w", lw=1)
        ax.axhline(epoch_len - 0.5, color="w", lw=1)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_depth_consistency(layers, layer_acc, layer_cos, position, save_path=None, title=None):
    """Left: per-layer decode accuracy of the coin at a fixed (retention) position.
    Right: cross-layer cosine of the coin direction. High off-diagonal cosine = a
    consistent direction across depth (refusal-like); low = depth-specific encoding."""
    layers = list(layers)
    layer_cos = np.asarray(layer_cos)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(layers, layer_acc, marker="o", color="C3", lw=2)
    axes[0].axhline(0.5, color="0.6", ls=":", lw=1, label="chance")
    axes[0].set_xlabel("layer (resid_post)")
    axes[0].set_ylabel("decode coin (accuracy)")
    axes[0].set_ylim(0.4, 1.03)
    axes[0].set_xticks(layers)
    axes[0].grid(alpha=0.2)
    axes[0].set_title(f"per-layer, position {position}", fontsize=10)
    axes[0].legend(fontsize=8)
    im = axes[1].imshow(layer_cos, origin="lower", cmap="viridis", vmin=-1, vmax=1)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_xlabel("layer"); axes[1].set_ylabel("layer")
    axes[1].set_title("direction cosine across depth", fontsize=10)
    axes[1].set_xticks(layers); axes[1].set_yticks(layers)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def _save(fig, save_path, dark=False):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fc = fig.get_facecolor() if dark else "white"
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fc)
