"""Generate the two replication notebooks with nbformat (run once: `python _build_notebooks.py`).

Keeping the notebooks generated from a script means they stay in sync with the
src/ API and are easy to regenerate; the committed .ipynb files are the artifact.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = """\
# Make src/ importable and use inline figures.
import sys; sys.path.insert(0, "../src")
%matplotlib inline
import numpy as np
from pathlib import Path
"""


def mess3_nb():
    cells = [
        new_markdown_cell(
            "# Belief-State Geometry in a Transformer's Residual Stream — Mess3\n"
            "\n"
            "Replication of the headline result of **Shai, Marzen, Teixeira, Gietelink "
            "Oldenziel & Riechers (2024)**, *Transformers Represent Belief State Geometry "
            "in their Residual Stream* ([arXiv:2405.15943](https://arxiv.org/abs/2405.15943)).\n"
            "\n"
            "**The claim.** A small transformer trained *only* to predict the next token of "
            "sequences from a Hidden Markov Model develops, in its residual stream, a "
            "*linearly decodable* representation of the optimal Bayesian **belief state** — "
            "the distribution over the HMM's **hidden states** given the observation history. "
            "For the **Mess3** process this belief lives in a 2-simplex (a triangle) and the "
            "set of reachable beliefs is an infinite **fractal**, predicted *in advance* from "
            "the data-generating process.\n"
            "\n"
            "**The crux (don't get this wrong).** The probe target is the belief over *hidden "
            "states*, **not** the next-token distribution. The next-token distribution is a "
            "(lossy) function of the belief; regressing onto it would replicate the wrong thing."
        ),
        new_code_cell(SETUP),
        new_markdown_cell(
            "## 1. Train (or load) the model\n"
            "Architecture matches the paper (App. A.4): 4 layers, d_model=64, 1 head, "
            "d_head=8, d_mlp=256, n_ctx=10, LayerNorm, ReLU, causal. We use a laptop-friendly "
            "Adam recipe that reaches the same loss floor as the paper's 1M-step SGD run; the "
            "deviation is documented in the README. Belief recovery is a property of the "
            "*converged* model, not of the optimiser."
        ),
        new_code_cell(
            "from hmms import Mess3\n"
            "from train import train, TrainConfig, save_checkpoint\n"
            "\n"
            "ckpt_path = Path('../checkpoints/mess3_fast.pt')\n"
            "if not ckpt_path.exists():\n"
            "    res = train(Mess3(), train_cfg=TrainConfig.fast(n_steps=10000, device='cpu'),\n"
            "                process_name='mess3')\n"
            "    save_checkpoint(res, ckpt_path)\n"
            "else:\n"
            "    print('Using existing checkpoint:', ckpt_path)"
        ),
        new_markdown_cell(
            "## 2. Probe the residual stream and build the figures\n"
            "`run_mess3_experiment` caches `blocks.{l}.hook_resid_post` at every position, "
            "fits a plain linear map (OLS) from the 64-dim activations to the 3-dim analytic "
            "belief, and evaluates R² on a **held-out** sample. It writes the figures and "
            "`results/metrics_mess3.json`."
        ),
        new_code_cell(
            "from experiments import run_mess3_experiment\n"
            "out = run_mess3_experiment('../checkpoints/mess3_fast.pt', save=True)\n"
            "m = out['metrics']\n"
            "print('final-layer R²      :', round(m['probe']['r2_headline_layer'], 4))\n"
            "print('R² per layer        :', {k: round(v,3) for k,v in m['probe']['r2_per_layer'].items()})\n"
            "print('final loss (nats)   :', round(m['training']['final_loss_nats'], 4))\n"
            "print('optimal floor (nats):', round(m['training']['optimal_in_context_loss_nats'], 4))"
        ),
        new_markdown_cell(
            "## 3. The headline figure\n"
            "**Left:** the analytic Mess3 belief fractal (optimal observer). **Right:** the "
            "belief decoded linearly from the residual stream. Both are coloured by the *true* "
            "belief (RGB = belief coordinates), so a visual match is a like-for-like result."
        ),
        new_code_cell("out['figures']['headline']"),
        new_markdown_cell("## 4. Training converged to the information-theoretic floor"),
        new_code_cell("out['figures']['training_curve']"),
        new_markdown_cell(
            "## 5. Which layer carries the geometry?\n"
            "For Mess3 the belief is recoverable from the final residual stream (the paper's "
            "finding). RRXOR — where the representation is *distributed across layers* — is in "
            "`02_rrxor_layers.ipynb`."
        ),
        new_code_cell("out['figures']['layer_r2']"),
    ]
    nb = new_notebook(cells=cells)
    return nb


def rrxor_nb():
    cells = [
        new_markdown_cell(
            "# Belief-State Geometry — RRXOR: distributed across layers, and beyond the next token\n"
            "\n"
            "The second result from [arXiv:2405.15943](https://arxiv.org/abs/2405.15943). "
            "**RRXOR** ('Random, Random, XOR') emits two fair bits then their XOR. Two things "
            "differ from Mess3:\n"
            "\n"
            "1. **The belief geometry is distributed across layers** — recovery from the "
            "*concatenation* of all layers' residuals beats any single layer.\n"
            "2. **The belief carries information about the whole future**, beyond the next "
            "token the model was trained on — the residual stream encodes belief that the "
            "next-token distribution (a many-to-one function of belief) cannot reconstruct."
        ),
        new_code_cell(SETUP),
        new_markdown_cell("## 1. Train (or load) the RRXOR model"),
        new_code_cell(
            "from hmms import RRXOR\n"
            "from train import train, TrainConfig, save_checkpoint\n"
            "\n"
            "# The committed checkpoint loads instantly. Delete it (or run the CLI with\n"
            "# --steps 50000) to retrain from scratch; RRXOR needs more steps than Mess3\n"
            "# to crystallise its (distributed) belief representation.\n"
            "ckpt_path = Path('../checkpoints/rrxor_fast.pt')\n"
            "if not ckpt_path.exists():\n"
            "    res = train(RRXOR(), train_cfg=TrainConfig.fast(n_steps=50000, device='cpu'),\n"
            "                process_name='rrxor')\n"
            "    save_checkpoint(res, ckpt_path)\n"
            "else:\n"
            "    print('Using existing checkpoint:', ckpt_path)"
        ),
        new_markdown_cell(
            "## 2. Probe per layer, across layers, and run the future-information analysis"
        ),
        new_code_cell(
            "from experiments import run_rrxor_experiment\n"
            "out = run_rrxor_experiment('../checkpoints/rrxor_fast.pt', save=True)\n"
            "m = out['metrics']\n"
            "print('R² per layer       :', {k: round(v,3) for k,v in m['probe']['r2_per_layer'].items()})\n"
            "print('R² concat layers   :', round(m['probe']['r2_concat_all_layers'], 4))\n"
            "print('concat beats single:', m['probe']['concat_beats_best_single_layer'])"
        ),
        new_markdown_cell(
            "## 3. The representation is distributed across layers\n"
            "The concatenation across layers (red) recovers the belief better than any single "
            "layer — the geometry is not localised to the final residual stream."
        ),
        new_code_cell("out['figures']['layer_r2']"),
        new_code_cell("out['figures']['pred_vs_true']"),
        new_markdown_cell(
            "## 4. The belief carries information beyond the next token\n"
            "**Left:** pairs of positions with nearly equal next-token distributions can still "
            "have very different beliefs (mass at the left edge) — the belief distinguishes "
            "futures the next token cannot. **Right:** the residual stream recovers the belief "
            "(high R²) far better than the next-token distribution can (the next-token "
            "distribution is a lossy function of the belief)."
        ),
        new_code_cell("out['figures']['future_information']"),
        new_code_cell(
            "print('R² residual → belief        :', round(m['future_information']['r2_residual_to_belief'], 4))\n"
            "print('R² next-token dist → belief :', round(m['future_information']['r2_nexttoken_dist_to_belief'], 4))"
        ),
    ]
    return new_notebook(cells=cells)


def main():
    for name, nb in [("01_mess3_replication.ipynb", mess3_nb()),
                     ("02_rrxor_layers.ipynb", rrxor_nb())]:
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                     "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        path = HERE / name
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
