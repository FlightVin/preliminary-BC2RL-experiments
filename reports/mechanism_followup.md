# BC2RL mechanism follow-up (amendment A6)

**Status: update 3, 2026-10-27 20:00 EDT.** Wave 1 training is complete. The workers report all 96 yf evaluation tasks done; look 1 is submitted (job 10960293) and has not run. Stage G (GSPO / CTPO, Section 9) is in validation. Update 1 added the exploratory results from the
existing A5 records (Y1) and from the development panels (yd). **No confirmation-panel (yf) outcome has been read.**
Every pre-registered test runs once, at look 1. Protocol: `protocol/AMENDMENT-A6-2026-10-26.md` (pre-registered at
commit `c6f7d0a`; dated additions 10.1.1-10.1.3 frozen before any yf read). The A5 study (`reports/extension_study.md`)
is complete. A6 does not change it. Section 14 of that report records five corrections from Y1.

## 0. Summary

| stage | question | state (2026-10-27 20:00) |
|---|---|---|
| Y1 interpretation audit | Is the A5 effect a larger change from a higher baseline, or a worse final state? Do matched counts hide different true probabilities? | done (exploratory, A5 records); Section 0.1 |
| Y2 truncation and failure types | Are intermittent solutions hidden by length or format? Which failure types make the extended-light difference? | dev done (exploratory); yf at look 1 |
| Y3 cross-difficulty | Do updates on 5- and 10-input problems cause the 3-input losses, or does 3-input training itself? | 24 of 24 branches trained and yf-evaluated; canary passed; look 1 submitted |
| Y4 BC exposure | Does BC exposure predict loss beyond the initial success distribution? Shape over steps 16-128? | 3 matched pairs frozen; all branches trained and yf-evaluated; look 1 submitted |
| method interaction | Does the objective (GRPO vs filtered) change the init difference? | tested at look 1 |
| Y5 E2 redesign | Can controlled single updates predict later losses? | decided after look 1 |
| G sequence-level and prefix objectives | Do GSPO or CTPO change coverage loss or its BC-init dependence, beyond length normalization and rollout reuse? | GPU validation pending (first attempt failed: script defect, fixed); Section 9 |

### 0.1 What the exploratory data show (not confirmation)

Y1 uses the A5 panel xc3 (768 problems, 24 initial and 24 final adapters). The dev results use yd3 (384 problems).
Both are exploratory. The same data generated the hypotheses in protocol 10.1.1. Look 1 tests them on yf3.

| finding | Y1 (xc3) | dev (yd3) |
|---|---|---|
| Delta (K = 16, cap 256), ext - light | +0.202, boot [0.175, 0.227], t [0.141, 0.262] | +0.281, boot [0.246, 0.316], t [0.204, 0.358] |
| Delta at cap 1,024 (subgroup fixed at cap 256) | not measured (A5 had cap 256 only) | +0.281 (identical; 0 rescued attempts) |
| Delta split: worse subgroup final state / better baseline | +0.130 / -0.071 | not computed on yd |
| Whole-panel finals, ext - light (pass@1 and pass@16) | ext better in 12/12 (method, seed) pairs | ext mean pass@1 0.475 vs light 0.388 |
| True p inside bins (NPMLE) | higher for ext in every bin | not computed on yd |
| K = 1 decomposition: illegal operand D_I | +0.342, t [0.044, 0.641] | +0.375, boot [0.350, 0.401], t [0.026, 0.725] |
| K = 1 decomposition: wrong target D_W | -0.144 | -0.132 (t-interval includes 0) |
| K = 1 decomposition: truncation D_T, format D_F | -0.012, +0.029 | -0.004, +0.031 (t-intervals include 0) |

- **R1 (truncation, format).** On yd, 33,491 of 737,280 attempts (4.5%) stopped at 256 tokens. After continuation of
  the same attempts to 1,024 tokens, **none** reached a valid solution. Of the 33,491 continued attempts, 47.8% are still
  truncated at 1,024 tokens, 42.3% end with an illegal operand, 7.9% are malformed and 2.0% have an arithmetic error. Five sampled texts show long chains of operations that repeat or
  drift (`slurm-agentic-engineering/results/bc2rl-a6-repro-main-s0/y4_repro.json`). Truncation is concentrated in a few adapters (for example
  b16-s66 12%, b64-s66 11%, b128-s33 9%); most adapters truncate < 1%. A lenient re-parse would accept 841 failed
  attempts (diagnostic only; never a reward). On yd, truncation up to 1,024 tokens does not explain the
  difference. Longer continuations were not measured. Look 1 tests this on yf.
- **R2 (comparison artifact).** "The extended start loses more on its own intermittent problems" is not "the extended
  start ends worse". On both panels the extended finals are better on the whole panel. Report Section 14 of
  `extension_study.md` corrects the A5 wording.
- **Failure types.** The largest component is illegal-operand failures. In Y1, online training from step 16 removed
  many of them on its subgroup (E_I from -0.19 to -0.64 per branch). This lowers the light loss and so
  increases Delta. This describes the measured difference. It is not a mechanism.
- **Problem structure (exploratory).** Losses concentrate on problems where every valid solution uses
  multiplication, where one solution form exists, and where the target is >= 100 (both steps; larger from step 128).
  The excess loss correlates with the drop in the policy's preference for the problem's best form (yd: r = 0.62 from
  step 128, 0.53 from step 16, pooled over branches). These are hypotheses H-S1-H-S3, frozen for look 1.

### 0.2 Initial competence by BC step (yd, 6 seeds each)

| BC step | yd3 pass@1 (mean, range) | yd5 pass@1 (mean) |
|---|---|---|
| 16 | 0.098 (0.069-0.127) | 0.005 |
| 32 (re-run path) | 0.151 (0.129-0.170) | 0.010 |
| 64 | 0.221 (0.207-0.237) | 0.015 |
| 128 | 0.284 (0.238-0.305) | 0.024 |

- **Matching (Section 8.2, frozen 10.1.2).** 3 of 216 cross-step adapter pairs meet all four criteria:
  b16-s44 / b32-s11, b64-s33 / b128-s66, b64-s44 / b128-s66. **No pair spans step 16 vs step 128.** The matched
  comparison can only test one doubling of BC exposure at matched competence. It cannot separate R5 from R6 for the
  full 16-to-128 contrast. The residual mismatch per pair is in protocol 10.1.2.
- **Reproduction check (Section 3.2, 10.1.3).** The re-run BC adapters are not bit-identical to the originals (0/18).
  Relative L2 distance: step 16 0.003-0.006, step 64 0.011-0.017, step 128 0.018-0.028. The re-run step-32 adapter is
  0.032 from the original step 16 and 0.041 from the original step 64. The yd pass@1 of the re-run adapters is queued
  (6 tasks). The step-32 point in the dose-response therefore comes from a BC path that differs from the original
  one (relative L2 0.003-0.006 at step 16). The report will state the pass@1 difference when it is measured.

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

## 7. Accounting and updated forecast (2026-10-27 19:58)

| item | value |
|---|---|
| BC2RL GPU-hours since job 10947200 (sacct, all attempts incl. 15 preempted; 31 GPU attempts, all `preempt`) | 75.6 (A6 75.2; stage G 0.4) |
| A6 worklist tasks complete | 72 of 73 (worker logs): all training; all 24 initial + 72 final yf evaluations; 5 of 6 reproduction yd evaluations |
| running | reproduction yd evaluation s66 (worker w15) |
| A6 wave-1 total (forecast) | about 77-78 GPU-h (update-2 forecast 72-78; pre-registered 75-93) |
| look 1 | one CPU job (10960293) after test job 10960288; structural check of the 96 yf evaluations first |

- Look 1 forecast: 2026-10-27 22:00 to 2026-10-28 06:00 EDT. The CPU queue sets the start. The update-2 forecast
  (17:00-20:00) was late: Slurm preempted worker w13 at 15:03 and 18:31; it waited about 3.7 h in the queue.
- Stage G forecast (protocol 12.7, 12.8): 105-155 GPU-h without T4F, 120-180 with T4F. Validation is now
  forecast at 1-1.5 GPU-h, not 0.3: model loading took about 17 min in the failed attempt.
- Slots: 5 BC2RL GPU slots, preempt only, counting pending jobs (`tools/bc2rl_guard.py`). No general GPU, no `ae-topup`.

## 8. Checks done in this update

- Y3 canary (`scripts/a6_canary.py`, job 10949310): 10 MASK/T3 runs. MASK used 0 completions from 5- and 10-input
  prompts while its stream sampled them in every round. T3 sampled only 3-input prompts. All 10 pass.
- Dev outputs recomputed outside the project code (laptop, from the harvested CSV/JSON): pass@1 profiles agree with
  the per-adapter table (48/48); the matching rule applied again to the 216 pairs gives the same 3 pairs; code counts
  sum to the attempt count.
- An independent statistical review of `scripts/y_look1.py` found 10 items (for example, the H-S3 claim could never be
  set). All were fixed or given a fixed reading before any yf outcome (protocol 10.1.4, commit b76839f).
- `scripts/y_look1.py` now has a synthetic end-to-end test (planted cross-difficulty effect; refusal of an incomplete
  set). The test job runs before look 1. Audit file: `slurm-agentic-engineering/audits/a6-update1-audit-20261027.md`.

## 9. Stage G status (GSPO / CTPO; protocol Section 12)

- Question (pre-registered 12.1, 12.8): do sequence-level weighting and clipping (GSPO) or cumulative-prefix
  correction (CTPO) change coverage loss or its dependence on BC initialization, beyond length normalization and
  rollout reuse? Arms C1, L1, L4, S4, T4, T4F (protocol 12.2, 12.8).
- Done: implementation; CPU test suite (96 passed, job 10956587); independent code review 1 (10 items, all
  resolved; audit `a6-stageG-code-review1-20261027.md`).
- GPU validation job 10956589 failed after 24 min. Cause: `scripts/g_validate.py` read the YAML config with PyYAML,
  which reads `1e-05` as a string. The training code uses the JSON-first loader and is not affected. Fix and
  regression test: commit 3f675ca. Rerun: job 10960289, after the test job. No stage-G result exists.
- Next: precision decision (V4), six unclipped probes, clip calibration on dev data, 12 dev runs, gate D.

## 10. Changes
- Update 0 (2026-10-26 ~19:45): design, matrix and forecast. No results.
- Update 1 (2026-10-27 04:16): Y1 and dev results (exploratory); matched pairs; reproduction check; canary;
  accounting and forecast. No yf outcome read.
- 2026-10-27 04:17 (scheduled pass): no new result. Timestamps of update 1 and protocol 10.1.2-10.1.4 corrected
  to the git commit times (they had been written as estimates 15-45 min late).
- Update 2 (2026-10-27 12:15): progress, accounting and forecast only. No new result; no yf outcome read; no
  analysis code changed. Workers w6-w9 ended normally (no failed task).
- Update 3 (2026-10-27 20:00): progress and accounting; stage G added (Section 9). Workers report all yf
  evaluations done; look 1 submitted. New script `scripts/a6_yf_accept.py` (structural check before look 1). No yf outcome read.
- 2026-10-27 20:01: timestamps of update 3 corrected to the git commit time (20:00; had been written as 20:20).
