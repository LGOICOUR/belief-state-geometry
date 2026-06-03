"""The linear probe: recover belief states from the residual stream.

This is the measurement that tests the paper's claim. We:

1. Run the trained model and cache the residual stream at each position
   (``blocks.{l}.hook_resid_post`` -- 64-dim vectors).
2. Fit a plain linear map (OLS, optionally Ridge) from those activations to the
   **analytic optimal belief over hidden states** -- NOT next-token logits. This
   distinction is the crux of the whole project: the belief is the distribution
   over the HMM's hidden states given the history; the next-token distribution is
   merely a function of it. Regressing onto next-token probabilities would
   "replicate" the wrong thing.
3. Report R^2 on a *held-out* activation set (fit and eval come from disjoint
   samples so the probe can't overfit), overall and per belief coordinate.

The fit is deliberately a low-capacity linear map (d_model -> n_states), so a
high R^2 means the belief really is *linearly* present in the residual stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

import beliefs as B
from data import EvalData


# --------------------------------------------------------------------- #
# Caching residual-stream activations
# --------------------------------------------------------------------- #
def cache_residuals(
    model,
    tokens: torch.Tensor,
    hook: str = "resid_post",
    layers=None,
    batch_size: int = 4096,
) -> dict[int, np.ndarray]:
    """Cache residual activations at every position, per layer.

    Parameters
    ----------
    hook: one of ``'resid_pre'``, ``'resid_mid'``, ``'resid_post'``.
    layers: iterable of layer indices (default: all layers).

    Returns ``{layer: X}`` with ``X`` of shape ``[n_seqs * seq_len, d_model]``
    (positions flattened in row-major order, matching ``beliefs.reshape(-1, S)``).
    """
    n_layers = model.cfg.n_layers
    layers = list(range(n_layers)) if layers is None else list(layers)
    names = [f"blocks.{l}.hook_{hook}" for l in layers]
    name_set = set(names)

    chunks: dict[int, list] = {l: [] for l in layers}
    model.eval()
    with torch.no_grad():
        for i in range(0, tokens.shape[0], batch_size):
            batch = tokens[i : i + batch_size]
            _, cache = model.run_with_cache(
                batch, names_filter=lambda n: n in name_set, return_type=None
            )
            for l in layers:
                a = cache[f"blocks.{l}.hook_{hook}"]      # [b, seq, d_model]
                chunks[l].append(a.reshape(-1, a.shape[-1]).cpu().numpy())
    return {l: np.concatenate(chunks[l], axis=0) for l in layers}


def flatten_beliefs(eval_data: EvalData) -> np.ndarray:
    """``[n_seqs, seq_len, n_states]`` -> ``[n_seqs*seq_len, n_states]`` (row-major)."""
    S = eval_data.beliefs.shape[-1]
    return eval_data.beliefs.reshape(-1, S)


# --------------------------------------------------------------------- #
# Fitting / evaluating the probe
# --------------------------------------------------------------------- #
@dataclass
class ProbeResult:
    layer: object               # int, or 'concat'
    hook: str
    r2: float                   # overall (uniform average across belief coords)
    r2_per_coord: np.ndarray    # [n_states]
    y_true: np.ndarray          # [N_test, n_states] analytic beliefs
    y_pred: np.ndarray          # [N_test, n_states] probe predictions
    n_fit: int
    n_test: int
    alpha: float
    estimator: object = None    # the fitted sklearn model (to apply to new data)

    def summary(self) -> str:
        coords = ", ".join(f"{v:.3f}" for v in self.r2_per_coord)
        return (f"layer={self.layer} hook={self.hook}: R^2={self.r2:.4f} "
                f"(per-coord [{coords}], n_fit={self.n_fit}, n_test={self.n_test})")


def _fit_eval(X_fit, Y_fit, X_test, Y_test, alpha=0.0):
    est = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
    est.fit(X_fit, Y_fit)
    pred = est.predict(X_test)
    r2 = r2_score(Y_test, pred)                                   # uniform average
    r2c = r2_score(Y_test, pred, multioutput="raw_values")       # per coordinate
    return est, float(r2), np.asarray(r2c), pred


def probe_all_layers(
    model,
    fit_data: EvalData,
    test_data: EvalData,
    hook: str = "resid_post",
    alpha: float = 0.0,
) -> dict[int, ProbeResult]:
    """Fit a separate probe at each layer; evaluate R^2 on held-out activations."""
    Xf = cache_residuals(model, fit_data.tokens, hook)
    Xt = cache_residuals(model, test_data.tokens, hook)
    Yf, Yt = flatten_beliefs(fit_data), flatten_beliefs(test_data)
    out = {}
    for l in Xf:
        est, r2, r2c, pred = _fit_eval(Xf[l], Yf, Xt[l], Yt, alpha)
        out[l] = ProbeResult(l, hook, r2, r2c, Yt, pred, len(Yf), len(Yt), alpha, est)
    return out


def probe_concat_layers(
    model,
    fit_data: EvalData,
    test_data: EvalData,
    hook: str = "resid_post",
    alpha: float = 0.0,
    layers=None,
) -> ProbeResult:
    """Fit one probe on residuals concatenated across layers.

    This is the key analysis for RRXOR, where belief geometry is *distributed*:
    recovery from the concatenation across layers beats any single layer.
    """
    Xf = cache_residuals(model, fit_data.tokens, hook, layers)
    Xt = cache_residuals(model, test_data.tokens, hook, layers)
    Xf_c = np.concatenate([Xf[l] for l in sorted(Xf)], axis=1)
    Xt_c = np.concatenate([Xt[l] for l in sorted(Xt)], axis=1)
    Yf, Yt = flatten_beliefs(fit_data), flatten_beliefs(test_data)
    est, r2, r2c, pred = _fit_eval(Xf_c, Yf, Xt_c, Yt, alpha)
    return ProbeResult("concat", hook, r2, r2c, Yt, pred, len(Yf), len(Yt), alpha, est)


def best_layer(results: dict[int, ProbeResult]) -> int:
    return max(results, key=lambda l: results[l].r2)


# --------------------------------------------------------------------- #
# Simplex coordinates for plotting (3-state beliefs)
# --------------------------------------------------------------------- #
def simplex_points(result: ProbeResult):
    """Project a 3-state ProbeResult to triangle coords for plotting.

    Returns ``(true_xy, pred_xy, rgb)`` where colours come from the *true* belief
    so the theory and recovered clouds are coloured identically.
    """
    true_xy = B.project_to_simplex_2d(result.y_true)
    pred_xy = B.project_to_simplex_2d(result.y_pred)
    rgb = B.belief_to_rgb(result.y_true)
    return true_xy, pred_xy, rgb
