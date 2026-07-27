# Phase 3 — the memory bottleneck: does an optimal predictor carry what it no longer needs?

## The question Phase 2 left open

Phase 2 showed a transformer keeps a predictively-defunct latent (the mixture coin)
linearly decodable at ~100% after it stops mattering — *super-sufficiency*. But there is a
confound baked into the architecture: the coin's indicator tokens stay inside the context
window, so attention can simply **re-read** them. Nothing has to survive a memory
bottleneck. The load-dependent forgetting of the retention ledger is real, but the
information is always in-principle re-readable.

So Phase 2 tests a transformer-specific claim, not a claim about optimal predictors in
general. Phase 3 removes the crutch.

## The bottleneck

Swap the transformer for a **recurrent model** (a GRU first; a state-space model in 3b). A
recurrent model has no attention: at position `t` it sees only the current token plus a
**fixed-size carried state** `h_t ∈ ℝ^{d_hidden}`. Anything from the past it wants at
position `t` must have been actively stored in `h_t`. Retaining the defunct coin now
**costs hidden-state dimensions** — so, unlike the transformer, an optimal recurrent
predictor has a reason to compress it away.

This is the clean form of the minimality question. Computational mechanics predicts the
minimal carried state is exactly the causal-state set (the ε-machine); a bottlenecked
optimal predictor should converge toward it.

## Prediction (informative either way)

- **Transformer (Phase 2):** retains the single defunct coin at *every* residual width —
  the capacity sweep proved width alone doesn't induce forgetting, because it re-reads.
- **Recurrent (Phase 3):** must carry the coin in `h_t`. Prediction: below some `d_hidden`
  threshold the GRU **drops** the defunct coin (minimality via *inaccessibility*, not load)
  while still tracking the live belief.
- **Headline:** retention-of-defunct-coin vs. `d_hidden`, showing a drop the transformer
  never shows. If it *doesn't* drop — super-sufficiency is architecture-general — that is
  also a real result.

## What we measure

1. **Belief geometry in a carried state** — probe `h_t` for the belief / generator marginal.
   Extends the Phase-1 result to a recurrent architecture (belief geometry isn't
   transformer-specific).
2. **The bottleneck curve** — retention(defunct coin) vs. `d_hidden` on the mixture (headline).
3. **The contrast** — transformer (re-read, retains at all widths) vs. recurrent (carried,
   drops below threshold), same process, same probe.
4. **Ablation carried over** — when the GRU *does* retain, is the retained coin still
   causally inert?

## Method notes

- Same processes, same probe methodology as Phase 1/2 — the only change is the substrate
  (`h_t` in place of `resid_post`). `RecurrentLM` (`src/rnn.py`) exposes the transformer's
  training interface, `model(tokens, return_type="loss")`, so the loss-floor comparison is
  identical, and `hidden_states(tokens)` for probing `h_t` like the residual.
- Loss floor: the same `optimal_in_context_loss` — an optimal predictor reaches it
  regardless of substrate. **P0's success criterion is the GRU converging to that floor.**
- The mixture process is unchanged; epoch-aligned sampling via `aligned_init_states`.

## Honest positioning

RNNs recovering ε-machines / causal states is prior territory (computational mechanics ↔
recurrent nets); cite, don't claim. The contribution is the project's throughline — the
**defunct-latent / super-sufficiency-under-bottleneck** framing and the
**transformer-vs-recurrent contrast**.

## Plan

- **P0** — GRU trains to the loss floor on `MixtureProcess(2,2)` (validates the substrate).
- **P1** — probe `h_t` for belief + coin-by-position (the retention probe on the carried state).
- **P2** — `d_hidden` sweep → the retention curve (headline).
- **P3** — transformer-vs-recurrent contrast + ablation.
- **3b** — a minimal state-space model (implement the recurrent scan by hand).
