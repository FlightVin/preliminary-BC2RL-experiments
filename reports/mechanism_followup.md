# BC2RL mechanism follow-up (amendment A6)

**Status: update 0, 2026-10-26 ~19:45 EDT. No A6 result exists yet.** This update gives the design, the experiment
matrix, the rationale, the timeline and the resource forecast. Protocol (pre-registered before any A6 job):
`protocol/AMENDMENT-A6-2026-10-26.md`, commit `c6f7d0a`. The A5 study (`reports/extension_study.md`) is complete. A6
does not change it.

## 0. Summary

| stage | question | state |
|---|---|---|
| Y1 interpretation audit | Is the A5 effect a larger change from a higher baseline, or a worse final state? Do matched counts hide different true probabilities? | CPU job 10947200 queued |
| Y2 truncation and failure types | Are intermittent solutions less likely, or hidden by length or format? Which failure types make the extended-light difference? | queued (workers) |
| Y3 cross-difficulty | Do updates on 5- and 10-input problems cause the 3-input losses, or does 3-input training itself? | queued (workers) |
| Y4 BC exposure | Does BC exposure predict loss beyond the initial success distribution? What is the shape over steps 16-128? | queued (workers) |
| method interaction | Does the objective (GRPO vs filtered) change the init difference? | tested at look 1 |
| Y5 E2 redesign | Can controlled single updates predict later losses? | decided after look 1 |

## 1. Starting point

A5 established one effect (X5, X7). On fresh 3-input problems that the initial adapter solved in 1-11 of 24 selection
attempts, 256 online rounds increase loss_16 = (1 - p)^16 more from BC step 128 than from BC step 16. The difference
was +0.203 (X5) and +0.200 (X7), with 12 of 12 (method, seed) pairs positive over seeds 11-66. A5 did not separate
explanations. Within every selection bin, the step-128 adapters had higher baseline success than the step-16 adapters.

## 2. Explanations and the stage that tests each

| id | explanation | test |
|---|---|---|
| R1 | Truncation at 256 tokens or format changes hide solutions that are still likely | Y2: same attempts at caps 256 and 1,024; failure codes; lenient re-parse |
| R2 | The effect is a larger change from a higher baseline, or matched counts hide different true p | Y1: final vs baseline split; NPMLE of p |
| R3 | Harder-problem updates damage 3-input solutions | Y3: M vs MASK |
| R4 | 3-input training itself concentrates probability on fewer solutions | Y3: MASK, T3 |
| R5 / R6 | BC exposure as such vs the initial success distribution it produces | Y4: dose-response; dev-matched pairs |
| R7 | The objective reinforces successes differently | method interaction test |

## 3. Experiment matrix (wave 1)

Seeds 11, 22, 33, 44, 55, 66 everywhere. GRPO, frozen settings (LR 1e-5, 32 prompts x 4 samples, cap 256, 256 rounds).

| arm | from BC step | stream | gradient | branches | status |
|---|---|---|---|---:|---|
| M (existing) | 16, 128 | mixed 3/5/10 inputs | all completions | 24 (GRPO + filtered) | reused |
| MASK | 16, 128 | identical to M | 3-input completions only, same loss scale | 12 | new |
| T3 | 16, 128 | 3-input prompts only | all (3-input) | 12 | new |
| T3E | 16, 128 | T3 at round 85 (same 3-input exposure as M) | — | 12 checkpoints of T3 | new |
| Y4 | 32 (re-run path), 64 | mixed (as M) | all | 12 | new |

- Each mixed stream holds exactly 2,732 3-input prompts (2 passes over the 1,366 3-input prompts of the 4,096-prompt
  pool). T3 holds 8,192. T3E holds 2,720.
- **Evaluation panels** (new, seed 20261027, disjoint from every earlier split and from the X5 panels):
  - development: yd3 = 384 3-input problems, yd5 = 96 5-input problems. They serve diagnosis and the Y4 matching
    decision.
  - confirmation: yf3 = 768 3-input problems, yf5 = 128 5-input problems. They serve every pre-registered test, in one
    look after all wave-1 adapters are evaluated.
- **Attempts:** initial adapters 48 (0-23 select, 24-47 baseline); finals 32. Every attempt has an outcome at cap 256
  and, by continuation of the same attempt, at cap 1,024.
- **Statistic:** the A5 estimator (bins 1-2 / 3-5 / 6-11 of 24, K = 16, pooled weights over the 24 initial adapters).
  A claim of a nonzero contrast needs both the 10,000-resample paired problem bootstrap and the seed-level t-interval
  (df = 5) to exclude 0.

## 4. Primary comparisons and predictions (from the protocol)

| comparison | R3 predicts | R4 predicts |
|---|---|---|
| C1 = Delta_M - Delta_MASK (cross-difficulty share of the init difference) | > 0, near Delta_M | near 0 |
| C2 = Delta_MASK (init difference without harder updates) | near 0 | > 0, near Delta_M |
| C3 = E_M - E_MASK per init | > 0 | near 0 |
| C4 = E_T3 - E_MASK per init | — | >= 0 |

- Y2-a: Delta at cap 1,024 with the subgroup fixed at cap 256. "Not explained by truncation" requires both lower limits
  > 0 and Delta_1024 >= 0.5 Delta_256.
- Y2-b: exact decomposition of Delta at K = 1 into failure codes (truncation, format, illegal operand, arithmetic,
  incomplete chain, wrong target).
- Y4: E at steps 16/32/64/128 with a pre-registered shape rule (gradual / threshold / non-monotonic / undetermined).
  Matched pairs come from dev measurements only: pass@1 within 0.02 on yd3 and yd5, a count histogram within total
  variation 0.10, and bin baselines within 0.03. If no pair matches, the report states "overlap inadequate".
- Method interaction: I = mean over seeds of Delta(filtered) - Delta(GRPO). A method claim needs both intervals.

## 5. Precision (planning)
The seed-level SD of Delta_s was about 0.06 in A5. For a difference of two Deltas over 6 seeds, the t-interval
half-width is about 0.08. Effects smaller than about half of Delta_M(GRPO) = 0.16 can stay undetermined. The report will
state the smallest effect that each interval excludes.

## 6. Timeline and forecast (from A5 throughput)

| item | work | GPU-h |
|---|---|---:|
| BC re-runs (steps 16/32/64/128) | 6 seeds | 0.6 |
| dev evaluations | 847,872 completions | 9-10 |
| Y3 + Y4 training | 36 GRPO branches (0.6-1.3 GPU-h each in X7) | 30-40 |
| confirmation evaluations | 3,096,576 completions | 33-37 |
| continuation to 1,024 tokens | truncated attempts only | 2-5 |
| **wave 1 total** | | **75-93** |

- Jobs: CPU prep 10947200 (tests, panels, Y1), then 5 preempt GPU workers 10947220, 10947233, 10947240, 10947242,
  10947245 (8 h wall each; each claims no new task after 4.5 h, and new workers continue the worklist).
- Wall time: about 17-21 h after the workers start, plus queue waits and preemptions. The order is: dev
  evaluations (hours 0-3), the dev analysis and Y4 matching (about hour 3), training branches with their confirmation
  evaluations (hours 2-17), and then look 1.
- Wave 2 (conditional): filtered-supervision Y3 (about 35 GPU-h), Y5. Each needs its own pre-registration.
- Resource rules: preempt GPUs only; at most 5 BC2RL GPU slots, counting pending jobs (`tools/bc2rl_guard.py`). GPU use
  before A6: 40.65 GPU-h.

## 7. Changes
- Update 0: design, matrix and forecast. No results.
