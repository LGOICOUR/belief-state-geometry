# Phase 2 — Belief-State Collapse in Mixture Processes

> **Status: built (P0–P5) and validated.** This document is the design/method spec;
> the construction, retention probe, horizon×seed sweep, direction/depth geometry,
> causal ablation, and capacity-pressure sweep are all implemented in `src/` and
> reproduced in `results/` (figures linked from the README). Headline: the residual is
> **super-sufficient** — it retains a predictively-defunct latent that the minimal
> belief discards, robustly (5 horizons × 3 seeds) and causally-inertly (ablation),
> and not removed by capacity pressure. Honest caveats are kept in §4 and §7.

## Abstract

Phase 1 established that a transformer trained only on next-token prediction
represents, linearly in its residual stream, the optimal Bayesian belief over a
generative process's hidden states — in the geometry predicted in advance from
the process. Phase 2 asks the natural follow-up: **is the predictive belief the
*whole* of what the model represents, or does it also carry generator identity
beyond predictive necessity?**

We build a family of **mixture processes** in which two mechanistically distinct
generators are *predictively identical* for a tunable number of steps `i` and
then diverge. The belief-state framework predicts that the two generators are
**collapsed onto the same belief geometry** until step `i`, then **bifurcate**.
We test whether the model reproduces that collapse-then-bifurcate structure with
the correct timing, and — the genuinely open question — whether it discards
generator identity the moment that identity stops being predictively useful.

---

## 0. The question

To predict optimally, a system need only represent the **minimal sufficient
statistic of the past for the future** — the belief state. Phase 1 confirmed the
model represents *at least* this. Phase 2 tests whether it represents *exactly*
this, or more:

- **Pure predictive sufficiency** — generators that are predictively equivalent
  are represented identically (collapsed), and predictively irrelevant
  information is discarded as soon as it becomes irrelevant.
- **Extra mechanistic memory** — the model tracks which generator is (or was)
  active even when that identity carries no predictive value.

These make *different* predictions about a quantity the model is free to keep or
discard, which is what makes this an experiment rather than a demonstration.

---

## 1. The process family (the construction)

A **mixture process** parameterized by a divergence horizon `i`. The stream is a
sequence of fixed-length **epochs** of length `i + tail` (default `tail = 2`).
Each epoch:

1. Draw a latent fair coin `Z ∈ {A, B}` (fresh each epoch).
2. Positions `1…i`: emit `i` uniform random bits (identical under A and B).
3. Positions `i+1 … i+tail` (the **divergent tail**): emit `Z`'s **indicator**
   bit, deterministically — `0` if `Z = A`, `1` if `Z = B`.

At the epoch boundary `Z` is resampled.

**Predictive-equivalence property.** For positions `1…i` the next-bit
distribution is uniform regardless of `Z`, so A and B are statistically
indistinguishable and `Z` is independent of the observed prefix. At position
`i + 1` (the first indicator) A emits `0` and B emits `1`: **predictive
equivalence breaks exactly at step `i + 1`.**

**Why `tail ≥ 2` and not a single reveal.** This is the one subtlety that
matters, and building the construction is what surfaces it. If the divergence
were a *single* token at `i + 1` that also ended the epoch, `Z` would become
predictively irrelevant the instant it is revealed — so the optimal belief would
never actually *commit* to it. There would be no regime where representing `Z`
is predictively *necessary*, only the question of whether the model *retains* an
already-useless fact, collapsing E1/E2 and E3 into one weaker test. A short
`Z`-dependent tail fixes this: during the tail, `Z` is both **known** (revealed
by the first indicator) and **still predictive** (it determines the remaining
indicators), so the belief legitimately commits. The epoch reset then makes `Z`
irrelevant, which is where the retention question (E3) lives. So `tail ≥ 2`
cleanly separates "represents the predictive belief" (E1/E2) from "retains
predictively-useless identity" (E3).

**As an HMM.** The hidden state is just `(Z, phase-in-epoch)` — the indicator is a
constant given `Z`, so no bits need to be remembered — giving `2(i + tail)`
states, built programmatically as a labeled transition tensor `T` and handed to
`Process`.

**Tunable knobs.**
- Horizon `i ∈ {1, 2, 5, 10, 20}` (the original project roadmap's set).
- `tail` (default 2): the length of the committed window.
- A harder variant can make the tail a non-trivial function of the prefix (e.g.
  copy-`b₁` then repeat it, or parity), but the constant indicator keeps task
  difficulty flat across `i`, isolating the horizon from learnability.

**Design choice that removes a confound.** *Epoch-aligned sequences + absolute
positions.* Seed every sequence at an epoch boundary; within-epoch phase is then
`position mod (i + tail)`, which the model reads from its positional embedding.
This removes the RRXOR-style synchronization problem so it cannot contaminate the
measurement.

---

## 2. What the optimal observer does (the analytic prediction)

Everything inherits from the existing belief machinery (`T` → `stationary`,
`belief_update`, `belief_trajectory`, MSP, entropy rate). The new target is the
**generator marginal**: group hidden states by their `Z` label and sum belief
mass to get `P(Z = A | history)`. By construction:

- **Positions `1…i`:** `P(Z = A) = ½` exactly — no evidence. The A- and B-branch
  belief points are **coincident** (collapsed).
- **Positions `i+1 … i+tail`:** the first indicator disambiguates → `P(Z = A)`
  jumps to `0` or `1` and **holds** through the tail. The branches **bifurcate**
  and stay separated while `Z` remains predictive.
- **Next epoch boundary:** `Z` is resampled, so the just-revealed `Z` becomes
  predictively irrelevant; the optimal belief resets to `½` and **discards** it.

This is verified in `tests/test_mixture.py`: for `i = tail = 2` the generator
marginal along an epoch is exactly `[½, ½, 1, ½]` (collapse, collapse, commit,
reset). It is a precise, tunable prediction — a **collapse → bifurcate → reset**
trajectory with the split at exactly `i + 1` — the dynamic analogue of Phase 1's
static Mess3 fractal.

---

## 3. The experiments

- **E1 — Bifurcation timing (core).** Probe `Z`-decodability from the residual as
  a function of within-epoch position. Prediction: at chance for `t ≤ i`, jumps
  high at `t = i + 1`. Sweep `i`; show the onset position tracks `i + 1`. Compare
  to the analytic onset.
- **E2 — Geometry (core).** Visualize the analytic belief manifold and the
  model-recovered one, colored by `Z`, showing the branches overlapping then
  splitting.
- **E3 — Retention of irrelevant information (the frontier).** After the reveal
  the model knows `Z`. At the next epoch boundary `Z` becomes useless. Does the
  model **drop** old `Z` exactly when it stops mattering (pure predictive
  sufficiency), or does a decodable **trace persist** (extra-mechanistic memory)?
- **E4 — Premature-distinctness control.** Is `Z` decodable *before* `i + 1`? It
  cannot be (see §4), so this is a **negative control** validating the probe — a
  reading above chance pre-reveal means the probe is leaking.

---

## 4. Honest accounting: entailed vs. empirical

This is what keeps the experiment from fooling itself. **Collapse before the
horizon is largely entailed, not discovered:** the model sees only emissions, and
`Z` is statistically independent of the first `i` emissions, so `Z` is
information-theoretically *absent* from the input until the reveal. No model
could decode it earlier. So E4 is a control, and "the branches are collapsed for
`t ≤ i`" is partly a theorem.

The **empirical content** lives in three places:

1. **Format & latency (E1):** the information arrives at `i + 1`, but is it
   represented *linearly* and *immediately*, or only after extra positions/layers?
2. **Geometry (E2):** does the model reproduce the *specific* bifurcation
   structure and timing across the whole `i` sweep?
3. **Retention (E3):** the one place the two hypotheses make *different*
   predictions, about something the model is free to keep or discard.

Stating this explicitly is the point: the result must not smuggle an entailed
fact in as a discovery.

---

## 5. Hypotheses and interpretation

| Outcome | Signature | Meaning |
|---|---|---|
| **Pure predictive sufficiency** | E1 onset exactly at `i + 1`; E3 old-`Z` dropped at the boundary | The representation *is* the predictive belief, nothing more — a clean, strong confirmation that predictive sufficiency is the complete account. |
| **Extra mechanistic memory** | E3 old-`Z` lingers after it is irrelevant | Models retain predictively-useless latent identity — the more consequential outcome, with direct interpretability/safety relevance (a model "knowing" things it has no current use for). |

The sharpest single contribution is **E3**: it is the one test where "represents
the predictive belief" and "represents the generator" diverge.

---

## 6. Controls and confounds

- **Learnability.** Verify the loss reaches the analytic floor for *every* `i`;
  report the gap. The constant `Z`-indicator keeps task difficulty flat across
  `i` (no parity to compute), so a missing split means collapse, not incapacity.
- **Context length.** `n_ctx ≥ i + 1`, ideally `≥ 2(i + 1)` to see the boundary
  in E3 — so `n_ctx` grows with `i` (≈ 42 for `i = 20`). Modest extra compute.
- **Probe discipline.** Held-out, linear; `Z` is binary, so report both decoding
  accuracy and R²; position-resolved (group activations by within-epoch position).
- **Probing ≠ causal use.** Optional stronger follow-up: ablate the decoded `Z`
  direction and check whether the reveal-bit prediction degrades — testing
  whether the model *uses* `Z`, not merely *stores* it.

---

## 7. Repo integration (no rewrites required)

The Phase 1 abstractions were shaped for exactly this:

- **`src/hmms/mixture.py`** — **implemented (P0):** `MixtureProcess(horizon, tail)`
  builds `T`; inherits `Process` → belief machinery for free; `generator_marginal`
  gives the `P(Z=A)` target. Covered by `tests/test_mixture.py`.
- **Generator target:** add a `state → Z` grouping and a `generator_marginal(belief)`
  helper; `make_eval_set` gains a `Z`-target alongside the hidden-state belief.
- **`src/probe.py`:** reuse; add a position-resolved variant and a binary `Z`
  probe (logistic, or regress the marginal). Everything else unchanged.
- **`src/data.py`:** add epoch-aligned seeding (one flag).
- **`src/experiments.py`:** `run_mixture_experiment(i)` → train/load → probe `Z`
  by position → figures → `metrics_mixture_i{N}.json`; loop over `i`.
- **`src/viz.py`:** add decodability-vs-position curves and the
  collapse→bifurcate manifold plot.
- **`tests/`:** `T` row-stochastic; `Z`-marginal `= ½` for `t ≤ i`; analytic
  bifurcation at `i + 1`; MSP finite.
- **`notebooks/03_mixture_collapse.ipynb`.**

---

## 8. Metrics and figures

**`results/metrics_mixture_i{N}.json`** per horizon: final loss vs floor,
`Z`-decodability per within-epoch position, onset position (model vs analytic),
the E3 retention metric, MSP size.

**Figures:**
- `Z`-decodability vs within-epoch position (one curve per `i`).
- onset position vs `i` (should track `i + 1`).
- belief manifold collapse → bifurcation (analytic vs recovered, colored by `Z`).
- retention across the epoch boundary (E3).

---

## 9. Milestones

- **P0** ✅ *(done)* — build + unit-test the mixture HMM (no ML).
- **P1** ✅ — generator-marginal target + analytic bifurcation prediction + tests.
- **P2** ✅ — train at a single `i`; loss reaches the floor.
- **P3** ✅ — position-resolved `Z` probe + direction/depth geometry.
- **P4** ✅ — horizon×seed sweep; onset tracks horizon; retention robust.
- **P5** ✅ — retention + causal ablation (inert) + capacity sweep (no emergent minimality).

---

## 10. Smallest first experiment (de-risk before the sweep)

`i = 2`, `tail = 2` (epoch length 4), `n_ctx = 8` (two epochs), one model.
Success criteria:

- training loss reaches the analytic floor (`0.75` bits `= (i+1)/(i+tail)`);
- `Z` undecodable for `t ≤ 2`, decodable from `t = 3` (the first indicator);
- the recovered generator marginal tracks the analytic `[½, ½, 1, ½]` per epoch.

If that holds, the `i`-sweep (P4) and the retention test (P5) are mechanical.
