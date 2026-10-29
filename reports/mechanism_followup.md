# BC2RL mechanism follow-up (amendment A6)

**Status: update 9, 2026-10-29 12:15 EDT.** New since update 8: the stage-G clip calibration (Section 10); the S2 build and gate S2 pass (Section 11.2); the R1 drift results (exploratory, Section 11.3); the first R3 measurements (one of two dev blocks done). Stage-G dev runs (3 of 12 done), S3 dev (canary) and the second R3 dev block are running. No confirmation outcome of stage G, S or R exists. Look 1 results are in Section 9 (update 5). Update 1 added the exploratory results from the
existing A5 records (Y1) and from the development panels (yd). **No confirmation-panel (yf) outcome has been read.**
Every pre-registered test runs once, at look 1. Protocol: `protocol/AMENDMENT-A6-2026-10-26.md` (pre-registered at
commit `c6f7d0a`; dated additions 10.1.1-10.1.3 frozen before any yf read). The A5 study (`reports/extension_study.md`)
is complete. A6 does not change it. Section 14 of that report records five corrections from Y1.

## 0. Summary

| stage | question | state (2026-10-28 12:14) |
|---|---|---|
| Y1 interpretation audit | Is the A5 effect a larger change from a higher baseline, or a worse final state? Do matched counts hide different true probabilities? | done (exploratory, A5 records); Section 0.1 |
| Y2 truncation and failure types | Are intermittent solutions hidden by length or format? Which failure types make the extended-light difference? | look 1: not explained by truncation; illegal-operand failures account for part of the difference (cap 256 only); Section 9.1 |
| Y3 cross-difficulty | Do updates on 5- and 10-input problems cause the 3-input losses, or does 3-input training itself? | look 1: the init difference arises without harder-problem updates (C2); their contribution is unresolved within the stated bounds, not absent (C1); Section 9.2 |
| Y4 BC exposure | Does BC exposure predict loss beyond the initial success distribution? Shape over steps 16-128? | look 1: gradual increase over steps 16-128; for 3 matched pairs the initial distribution explains the step effect (problem interval only); Section 9.4 |
| method interaction | Does the objective (GRPO vs filtered) change the init difference? | look 1: not shown (seed interval includes 0); Section 9.3 |
| Y5 E2 redesign | Can controlled single updates predict later losses? | not run (protocol 13.2; Section 9.7) |
| G sequence-level and prefix objectives | Do GSPO or CTPO change coverage loss or its BC-init dependence, beyond length normalization and rollout reuse? | validation passed (V3, V7); BF16 too noisy (V4) -> FP32 training passes; probes queued; Section 10 |

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

## 7. Accounting and updated forecast (2026-10-29 12:14)

Update 9: BC2RL used **100.4 GPU-hours** since job 10947200 (`sacct -D -X`, all attempts, BC2RL job names only,
running jobs up to 12:14; 46 GPU rows, all `preempt`). Since update 8: +23.4 GPU-h. By stage: A6 76.2; stage G 5.6
(probes, calibration, dev); R1 7.9; R3 8.1; S2 1.3; S3 1.3. BC2RL held 5 GPU slots (the cap) from 07:10, except
for gaps of a few minutes between the end of a job and the next submission. The table
below is from update 6.

| item | value |
|---|---|
| BC2RL GPU-hours since job 10947200 (sacct, all attempts incl. 16 preempted; 33 GPU attempts, all `preempt`) | 76.7 (A6 76.2; stage G 0.5) |
| A6 wave 1 | complete: 76.2 GPU-h (pre-registered forecast 75-93) |
| stage G validation | 0.1 GPU-h (job 10967595, 6 min); the failed first attempt used 0.4 GPU-h |
| stage G probes | canary worker 10968782 pending (Priority); second worker after the canary's first training line |

- Stage G forecast: the 12.7 / 12.8 ranges (105-155 GPU-h without T4F, 120-180 with T4F) assumed BF16 passes. FP32
  gradient passes take 1.8-1.9x the BF16 time (12.10). Sampling stays BF16, so the run-level increase is smaller;
  its size is unknown until the probes report round times. The revised forecast comes after the probes. Until then,
  treat the ranges above as lower bounds for the K = 4 arms.
- Wave-2 forecast (A6): none. The 9.7 decisions are made (protocol 13.2): no wave-2 runs.
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

- **Reproduction check (Section 3.2, protocol 10.1.3; yd3, exploratory; not used in any test).** pass@1 of the
  re-run BC adapters minus the originals, 6 seeds per step:

  | BC step | mean difference | largest absolute difference | seed SD of originals |
  |---|---:|---:|---:|
  | 16 | +0.0014 | 0.0052 | 0.020 |
  | 64 | -0.0047 | 0.0100 | 0.012 |
  | 128 | -0.0031 | 0.0204 (s22) | 0.028 |

  At each step the mean and the largest difference are smaller than the seed SD of the originals. The BC path is
  not bit-reproducible (10.1.3). The step-32 adapters come from the re-run, and step 32 has no original. This check
  estimates, but does not bound, the effect of mixing re-run and original adapters in Y4.

- **Look-1 checks (update 5).** Structural acceptance (`scripts/a6_yf_accept.py`, phase 1 of job 10967586): 192 of
  192 tag-panel rows pass (row counts, attempt counts, adapter hashes for the 42 registered adapters, one panel hash,
  record files). The hash check covers the 42 adapters registered in `configs/a6/adapter_sha256.json`; for the
  other 54 adapters it checks only that both panels have the same hash. Independent recompute
  (`slurm-agentic-engineering/audits/a6_look1_recompute.py`, own code from the protocol 3.4 definitions, input =
  per-problem yf3 counts exported by job 10968369): 15 of 15 statistics agree. Points and t-intervals agree to
  < 1e-4 (the project code uses the rounded t quantile 2.571); bootstrap limits agree to < 0.001 (independent
  resamples; largest difference 0.0009). The recompute checks the estimator only: the counts, the cap-1,024
  continuation and the failure codes come from the project evaluation output.

## 9. Look 1 results (confirmation panels yf3 / yf5; job 10967586, 2026-10-28 10:29 EDT)

Look 1 ran once, at commit c2f55bd, after a structural check of all 96 yf evaluations (Section 8). An earlier look-1
job (10960293, commit c6e7339) was cancelled while pending and never ran (empty log). The rule for a claim is the
protocol rule (3.4): both the paired problem bootstrap (10,000 resamples) and the seed-level t-interval (df = 5)
exclude 0. The problem interval is conditional on the trained adapters; only the seed interval covers variation
over training runs. This report applies no correction for the number of tests. Fifteen of the statistics (Delta_M,
C1, C2, the method interaction, Y2-a at both caps, the four Y4 E values and three increments, H-W1 at pass@1 and
pass@16) were recomputed from the per-problem counts outside the project code (Section 8). The others come from the
project code only (`scripts/y_look1.py`, reviewed before look 1; update-1 audit).

### 9.1 Y2 truncation and failure types (12 pairs; R1)
| statistic | point | problem bootstrap 95% | seed t 95% | reading |
|---|---:|---|---|---|
| Delta at cap 256 | +0.185 | [+0.158, +0.211] | [+0.120, +0.249] | — |
| Delta at cap 1,024 | +0.185 | [+0.158, +0.211] | [+0.120, +0.249] | **not explained by truncation** |
| Delta_256 - Delta_1024 | 0.000 | [0, 0] | [0, 0] | — |

- Among the 36 Y2 adapters (12 initial, 24 M finals), no problem gained a success from continuation to 1,024 tokens,
  so Delta is identical at the two caps for every seed. (In all 96 yf3 evaluations, one problem of one Y4 final,
  `b64-grpo-M-s11`, went from 17 to 18 successes.) The development panels gave the same result (10.1.2). R1
  (truncation) does not explain this effect.
- Failure codes at K = 1. Delta(K = 1) = +0.210 at both caps (both intervals > 0). D_k is the contribution of code k
  (extended minus light excess rate); the shares sum to 100%.

| code | D_k cap 256 | problem bootstrap | seed t | share | D_k cap 1,024 | problem bootstrap | seed t | share |
|---|---:|---|---|---:|---:|---|---|---:|
| illegal operand (I) | +0.328 | [+0.307, +0.349] | [+0.025, +0.631] | 157% | +0.304 | [+0.282, +0.326] | [-0.008, +0.616] | 145% |
| format (F) | +0.031 | [+0.027, +0.036] | [-0.006, +0.069] | 15% | +0.032 | [+0.027, +0.036] | [-0.009, +0.072] | 15% |
| truncation (T) | -0.012 | [-0.020, -0.005] | [-0.121, +0.097] | -6% | +0.013 | [+0.009, +0.016] | [-0.067, +0.092] | 6% |
| arithmetic (A) | -0.002 | [-0.010, +0.004] | [-0.036, +0.031] | -1% | -0.003 | [-0.010, +0.004] | [-0.037, +0.031] | -1% |
| incomplete chain (P) | +0.002 | [-0.001, +0.004] | [-0.003, +0.006] | 1% | +0.002 | [-0.001, +0.004] | [-0.003, +0.006] | 1% |
| wrong target (W) | -0.138 | [-0.149, -0.127] | [-0.479, +0.204] | -66% | -0.138 | [-0.149, -0.127] | [-0.479, +0.204] | -66% |

- At cap 256, only illegal-operand failures meet the rule ("accounts for part of the difference"). At cap 1,024, no
  code meets it (D_I seed interval [-0.008, +0.616]). From cap 256 to 1,024, D_I falls by 0.024 and D_T rises by 0.025;
  the codes are differences of excess rates, so this does not trace individual attempts.
- Code attribution is uncertain. The seed values of D_I range from +0.045 to +0.840, and at seed 11 D_I = +0.840 and
  D_W = -0.657 nearly cancel. This pattern is consistent with failures moving between the illegal-operand and
  wrong-target codes, not with two separate effects. The illegal-operand share (157%) is therefore not a precise size.

### 9.2 Y3 cross-difficulty (GRPO, 6 seeds; R3 vs R4)
| contrast (K = 16, cap 256) | point | problem bootstrap | seed t | reading |
|---|---:|---|---|---|
| Delta_M (reference) | +0.156 | [+0.126, +0.185] | [+0.083, +0.229] | — |
| **C1** = Delta_M - Delta_MASK | +0.009 | [-0.008, +0.024] | [-0.078, +0.095] | not shown |
| **C2** = Delta_MASK | +0.147 | [+0.120, +0.173] | [+0.089, +0.206] | **holds** |
| C3 step 16 (E_M - E_MASK) | +0.009 | [-0.003, +0.022] | [-0.042, +0.060] | not shown |
| C3 step 128 | +0.018 | [+0.007, +0.028] | [-0.030, +0.065] | not shown (problem interval only > 0) |
| C4 step 16 (E_T3 - E_MASK) | -0.018 | [-0.032, -0.003] | [-0.053, +0.018] | not shown (problem interval only < 0) |
| C4 step 128 | +0.028 | [+0.015, +0.042] | [-0.050, +0.106] | not shown (problem interval only > 0) |
| Delta_T3 | +0.193 | [+0.165, +0.220] | [+0.114, +0.273] | > 0 |
| Delta_T3E | +0.159 | [+0.130, +0.187] | [+0.107, +0.211] | > 0 |
| Delta_MASK - Delta_T3E (secondary) | -0.012 | [-0.029, +0.006] | [-0.085, +0.062] | not shown |
| step 16 E_MASK - E_T3E (secondary) | +0.045 | [+0.031, +0.058] | [+0.009, +0.082] | > 0 |
| step 128 E_MASK - E_T3E (secondary) | +0.033 | [+0.022, +0.045] | [-0.042, +0.109] | not shown |

- **The init difference arises without harder-problem updates** (C2 > 0, both intervals). C2 is 94% of Delta_M,
  which matches the R4 prediction "C2 > 0, near Delta_M".
- **A contribution of harder-problem updates is not shown** (C1). This is not evidence of no contribution: the seed
  interval allows C1 up to +0.095 (61% of Delta_M). Conditional on these adapters, the problem interval excludes
  C1 > +0.024. The planning note (protocol 6.4) expected this precision limit.
- Other K and caps (secondary). C2 > 0 on both intervals at K = 1, 4 and 24 and at cap 1,024 (cap-1,024 values are
  identical to cap 256); C1 is not shown in any of them. Three secondary results go against a pre-registered
  prediction:
  - C4 < 0 on both intervals at K = 1 (step 16: -0.091, seed t [-0.126, -0.056]; step 128: -0.093, [-0.118, -0.069])
    and at K = 4, step 16 (-0.051, [-0.080, -0.022]). R4 predicts C4 >= 0. At K = 1 this means more 3-input exposure
    (T3) gives lower single-attempt failure than MASK; at K = 16 the C4 sign is not decided.
  - C3 < 0 on both intervals at K = 1, step 128 (-0.026, [-0.044, -0.007]). R3 predicts C3 > 0.
  - At step 16, MASK loses more than T3E at K = 16 and 24 (both intervals) but less at K = 1 (-0.031, [-0.050,
    -0.011]). T3E has equal 3-input exposure, fewer optimizer steps and no harder rollouts.
- Manipulation checks. MASK and T3 used 0 completions from 5- or 10-input prompts (24 of 24 branches with logs). The
  check for M (5-input terms present) is **not available**: the 12 M branches (from A5) have no `used_by_inputs` log.
  The yf5 pass@1 change (final - initial, mean over seeds) does not separate the arms by harder-problem gradient:

  | arm | step 16 | step 128 |
  |---|---:|---:|
  | M | +0.036 | +0.049 |
  | MASK | +0.012 | +0.008 |
  | T3 | +0.014 | +0.047 |
  | T3E | +0.010 | +0.020 |

  T3 (no harder rollouts) gains as much on yf5 as M at step 128.
- Displacement from the initial adapter (L2, mean over seeds; diagnostic):

  | step | arm | round 64 | round 128 | round 256 |
  |---|---|---:|---:|---:|
  | 16 | M | 0.33 | 0.47 | 0.64 |
  | 16 | MASK | 0.33 | 0.46 | 0.62 |
  | 16 | T3 | 0.39 | 0.54 | 0.75 |
  | 128 | M | 0.30 | 0.41 | 0.59 |
  | 128 | MASK | 0.30 | 0.39 | 0.57 |
  | 128 | T3 | 0.32 | 0.46 | 0.65 |

  MASK moves the adapter almost as far as M at every round.

### 9.3 Method interaction (R7)
I = mean_s [Delta(filtered) - Delta(GRPO)] = +0.058; problem bootstrap [+0.041, +0.076]; seed t [-0.008, +0.124]
(5 of 6 seeds > 0). Conditional on these adapters, the problem interval excludes 0. The seed interval does not, so
**no method difference is shown** under the Section 7 rule.

### 9.4 Y4 BC exposure (GRPO; R5 vs R6)
| step | E | problem bootstrap | seed t |
|---|---:|---|---|
| 16 | +0.163 | [+0.135, +0.192] | [+0.106, +0.220] |
| 32 | +0.208 | [+0.185, +0.232] | [+0.180, +0.235] |
| 64 | +0.251 | [+0.223, +0.279] | [+0.221, +0.281] |
| 128 | +0.319 | [+0.291, +0.346] | [+0.270, +0.367] |

| increment | point | problem bootstrap | seed t |
|---|---:|---|---|
| I1 = E32 - E16 | +0.045 | [+0.022, +0.067] | [-0.007, +0.097] |
| I2 = E64 - E32 | +0.043 | [+0.022, +0.065] | [+0.016, +0.071] |
| I3 = E128 - E64 | +0.068 | [+0.049, +0.086] | [+0.035, +0.100] |

- Shape: **gradual** (all increments > 0; I2 and I3 have both lower limits > 0). The step-32 adapters come from the
  re-run BC path (10.1.3; reproduction check in Section 8).
- Matched pairs (10.1.2; residual dev mismatch as recorded there and in `look1.json`):

  | pair | D | d pass@1 yd3 | d pass@1 yd5 | TV | max d bin baseline |
  |---|---:|---:|---:|---:|---:|
  | b16-s44 / b32-s11 | -0.031 | 0.013 | 0.003 | 0.086 | 0.020 |
  | b64-s33 / b128-s66 | +0.003 | 0.011 | 0.004 | 0.078 | 0.010 |
  | b64-s44 / b128-s66 | -0.016 | 0.001 | 0.003 | 0.089 | 0.027 |

  Mean D = -0.014, problem bootstrap [-0.040, +0.011]; not all D > 0. The interval is within +-0.05, so by the
  protocol rule (8.3) **the matched initial distribution explains the step effect for these pairs (problem
  interval only)**. BC exposure beyond the measured initial distribution is not shown.
- Limits of the matched result: it has no seed-level interval (two pairs share b128-s66), and the seed SD of a single
  branch E is 0.026-0.054, about as large as the +-0.05 band, so training-run variation is not in the interval. There
  are 3 pairs, and they cover only 16 vs 32 and 64 vs 128 (no 16 vs 128 pair). Conditional on these adapters, the
  interval excludes a mean D above +0.011, which is smaller than one observed increment (0.043-0.068).
- Regression of E on log2(step) and yd3 pass@1 (24 checkpoints; descriptive only, r = 0.968 between the
  predictors): coefficient 0.005 for log2(step), 0.728 for yd3 pass@1.

### 9.5 Hypotheses frozen from Y1 (10.1.1; M finals, both methods)
| hypothesis | point | problem bootstrap | seed t | result |
|---|---:|---|---|---|
| H-W1 whole-panel final pass@16, extended - light | +0.129 | [+0.112, +0.147] | [+0.086, +0.172] | **holds** |
| H-W1 whole-panel final pass@1, extended - light | +0.113 | [+0.098, +0.128] | [+0.096, +0.130] | **holds** |
| H-F1 E_I step 16 < 0 | -0.356 | [-0.376, -0.335] | [-0.481, -0.231] | **holds** |
| H-S1 step 16 (needs-multiplication minus other) | +0.310 | [+0.267, +0.354] | [+0.237, +0.384] | **holds** |
| H-S1 step 128 | +0.368 | [+0.328, +0.408] | [+0.329, +0.407] | **holds** |
| H-S2 (H-S1 difference, step 128 - step 16) | +0.058 | [+0.017, +0.097] | [-0.013, +0.128] | not shown |
| H-S3 step 16 (mean correlation, 12 branches) | +0.535 | — | [+0.468, +0.601] | **holds** (t only, 10.1.4) |
| H-S3 step 128 | +0.600 | — | [+0.549, +0.652] | **holds** (t only) |

- H-W1: the extended-init finals are better on the whole panel at pass@1 and pass@16. A larger subgroup loss
  does not mean a worse final state on the panel.
- H-S1 and H-S3 are associations on the subgroup. H-S3 uses the panel-wide share of each solution form among all
  successes, so a problem's "preference drop" depends only on which forms the problem admits. H-S3 therefore
  overlaps the feature association of H-S1; it does not measure a per-problem strategy shift, and it does not show
  that the drop causes the loss. On the whole yf3 panel, 473 of 768 problems need multiplication. The init
  difference is not shown to depend on this feature (H-S2).
- Also reported: E_I step 128 = -0.028 and E_W at both steps; their seed intervals include 0.

### 9.6 What look 1 establishes and what it does not
- Established (both intervals): the effect is not a truncation artifact; it arises when GRPO updates use 3-input
  completions only (C2); it grows gradually with BC exposure from step 16 to step 128; the extended-init finals have
  higher whole-panel pass@1 and pass@16.
- Problem interval only: for the 3 dev-matched pairs, the initial success distribution accounts for the step effect.
- Not established: whether harder-problem updates add to the effect (C1; seed interval up to +0.095); a method
  difference (seed interval includes 0); whether BC exposure matters beyond the initial distribution (no 16 vs 128
  match; no seed-level interval for the matched pairs); any causal mechanism behind H-S1 or H-S3.
- Explanation status: R1 rejected for this effect. R4 (same-difficulty concentration) is consistent with C2, but its
  secondary prediction C4 >= 0 fails at K = 1 and K = 4 (step 16). R3 is not excluded at the seed level, and its
  secondary prediction C3 > 0 fails at K = 1, step 128. R6 is consistent for the matched pairs (problem interval
  only); R5 is not shown. R7 is not shown. Generality beyond this task, model and training budget is not tested.

### 9.7 Wave 2 decisions (protocol 6.5, 9, 10)
- Filtered Y3 (6.5): condition (a) holds (C2 decided). Condition (b) needs a live objective-dependence question from
  the method interaction or the per-method A5 values. The method interaction is not a Section 7 claim; its problem
  interval excludes 0 and 5 of 6 seeds are > 0, which is a judgement input only. Decision pending; it will be a dated
  addition before any run.
- Y5 E2 redesign (protocol 9): it runs only if it can resolve a question that remains after look 1, with the
  required pre-registration content; otherwise the report gives the reason. Decision pending.
- Stop rule (10): the main remaining question is the size of C1, which 6 seeds leave between about -0.08 and +0.10.
  The decision records will state whether any wave-2 option can narrow it.
- **Decided 2026-10-28 17:22 (protocol 13.2).** C1 needs about 20 seeds for the planned precision (Section 11.1).
  The rule adds seeds only if 12 or fewer are needed. Therefore no Y3 seeds are added, and filtered Y3 and the Y5
  E2 redesign do not run. The program-S masked-transfer arms (S3) test transfer directly at lower cost.
- **Owner confirmation 2026-10-29 (A6-R2).** The owner kept this decision. The report states C1 as unresolved within
  the stated bounds, not absent. The new contrasts (Section 11.1) stay exploratory. The two "Decision pending"
  bullets above are closed by this decision.

## 10. Stage G status (GSPO / CTPO; protocol Section 12)

- Question (pre-registered 12.1, 12.8): do sequence-level weighting and clipping (GSPO) or cumulative-prefix
  correction (CTPO) change coverage loss or its dependence on BC initialization, beyond length normalization and
  rollout reuse? Arms C1, L1, L4, S4, T4, T4F (protocol 12.2, 12.8).
- Done: implementation; CPU test suite (96 passed, job 10956587); independent code review 1 (10 items, all
  resolved; audit `a6-stageG-code-review1-20261027.md`).
- GPU validation job 10956589 failed after 24 min. Cause: `scripts/g_validate.py` read the YAML config with PyYAML,
  which reads `1e-05` as a string. The training code uses the JSON-first loader and is not affected. Fix and
  regression test: commit 3f675ca. Reruns 10958575 and 10960289 were cancelled while pending (replaced; never ran).
- Validation (job 10967595, protocol 12.10): V3 passes (GSPO and CTPO gradients equal length-normalized GRPO at the
  first update: relative L2 <= 1e-5, cosine 1.0). V7 passes (BF16 repeat passes identical). V4 fails for BF16: after
  one step, BF16 noise is 48-78% of the FP32 sequence log-ratio (limit 10%). **All stage-G arms train with FP32
  passes**; sampling stays BF16.
- The measured one-step sequence log-ratio (median 0.004-0.008) is about 10-27 times the paper's GSPO clip band
  (3e-4 / 4e-4). Clip calibration on the probes (12.4) sets the bands; no clipping result exists yet.
- **Clip calibration (12.11, recorded 07:18 before any dev run; job 10978094).** The rule of 12.4 / 12.8 gives
  e_tok = 0.011 and e_seq = 0.016. CTPO: T4 (p = 0.5) e_lo 0.084, e_hi = e_c 0.17; T4F (p = 0) 0.74 / 1.5. No value
  was chosen by judgement. At the paper's sequence band (3e-4 / 4e-4), 92-95% of probe sequences fall outside; the
  paper band would clip most sequences in this setting. This is a measured property of the probes, not a result on
  coverage.
- **Dev runs (12 configs, FP32 passes; tests 119 passed, job 10978119).** Done at 12:14: light C1, L1, L4 (57-63
  min each). Running: light S4 and T4. Measured round times: light C1 / L1 about 12 s; light S4 about 20 s. Extended
  runs have no round time yet. No dev analysis exists; gate D is unchanged.
- Forecast (planning only): the remaining 9 dev runs need about 9-14 GPU-h at the light rates; extended runs are not
  measured. The 12.7 / 12.8 confirmation ranges stay lower bounds until gate D.
- Next: 12 dev runs + yd finals -> dev analysis -> gate D.

## 11. Program S and stage R (amendment Sections 13 and 14)

### 11.1 S1: cross-difficulty contrasts per initialization (exploratory)
Source: `audits/s1_cross_difficulty.py` over the look-1 yf3 counts (job 10967586 export). It uses the Section 3.4
estimator and reproduces all 8 look-1 E points. These contrasts were not pre-registered at `c6f7d0a`. They reuse the
opened confirmation panel. They are descriptive; there is no multiplicity correction (12 rows at K = 16; 9 shown; T3 - MASK is in the JSON).
E = excess coverage loss at K = 16; a positive contrast means more loss in the first arm.

| contrast | init | point | seed t95 (df 5) | problem boot95 | seeds > 0 | seed SD | seeds needed |
|---|---|---|---|---|---|---|---|
| harder updates (M - MASK) | b16 | +0.009 | [-0.042, +0.060] | [-0.003, +0.021] | 4/6 | 0.049 | 9 |
| | b128 | +0.018 | [-0.030, +0.065] | [+0.007, +0.028] | 4/6 | 0.045 | 8 |
| | b128 - b16 (C1) | +0.009 | [-0.078, +0.095] | [-0.008, +0.024] | 3/6 | 0.083 | 20 |
| equal total samples (M - T3) | b16 | +0.027 | [+0.004, +0.050] | [+0.012, +0.042] | 6/6 | 0.022 | 4 |
| | b128 | -0.011 | [-0.057, +0.035] | [-0.022, +0.001] | 2/6 | 0.044 | 8 |
| | b128 - b16 | -0.037 | [-0.091, +0.016] | [-0.056, -0.019] | 1/6 | 0.051 | 10 |
| equal 3-input exposure (MASK - T3E) | b16 | +0.045 | [+0.009, +0.082] | [+0.032, +0.058] | 5/6 | 0.035 | 6 |
| | b128 | +0.033 | [-0.042, +0.109] | [+0.022, +0.045] | 3/6 | 0.072 | 16 |
| | b128 - b16 | -0.012 | [-0.085, +0.062] | [-0.029, +0.005] | 3/6 | 0.070 | 15 |

- "Seeds needed" = the smallest n with t(0.975, n-1) x SD / sqrt(n) <= 0.039 (= 0.25 x Delta_M; protocol 13.2).
- No harder-update contrast has both intervals above 0. The data do not show that the 5-/10-input updates harm
  3-input coverage, and they do not exclude an effect of about +0.06 per init.
- Two b16 contrasts have both intervals above 0. MASK - T3E also differs in optimizer steps and harder rollouts
  (T3E has fewer of both). It does not isolate one cause.
- No init difference has both intervals excluding 0. The cross-difficulty question stays unresolved at 6 seeds.

### 11.2 Program S and stage R status (2026-10-29 12:14)
- Protocol: Section 13 (program S), Section 14 (stage R) and the dated additions 13.11.1, 13.11.2 parts 1 and 2,
  14.9.1 and 14.9.2. Each was committed before the jobs that it governs.
- **S2 (13.11.2; selection data only, no outcome).** Class is confounded with the target: all 958 class-A pool
  candidates have a target in 1-60, but only 245 of 958 class-B candidates do. The first CEM gave 178 pairs (max
  2N = 128). The added rule (part 1) prescreened every remaining class-B pool candidate in that bin (1,175). CEM
  over 3,091 candidates gave **650 matched pairs**. **Gate S2: pass; N = 256** (512 problems and 16 passes per
  problem in every arm).
- **S2 limitation.** The matched pools still differ (A vs B): target 14.0 vs 36.2; solution traces 4.08 vs 1.04;
  prescreen success light 0.086 vs 0.072, extended 0.254 vs 0.190. S3 contrasts are within class and do not remove
  these differences. A difference between the A and B arms is therefore not attributable to class alone.
- **S3 dev (13.4; seed 11; 10 runs + 2 initial evaluations).** Tests passed (121, job 10978744). Canary 10979276:
  the online step of the light A run took 1,431 s for 256 rounds (logged rounds about 3.3 s; mean response about 26
  tokens). The other
  S3 workers start when a slot is free. No S3 analysis exists.
- **R3 dev (14.9.2).** rprep: pool 256 (334 of 1,024 screened problems qualified; 255 common to both inits), 1.24
  GPU-h. Block b128-s7: 256 rounds in every arm (S, I, F, O), masked fraction 0.019 (limit 0.05), losses finite;
  10,321 s (2.9 GPU-h). Block b16-s7: round 224 of 256 at 12:10; the last three logged rounds took 48-77 s. The dev evaluations
  (rd3, yd3) follow each block. Gate DR is not yet decided.
- **Revised stage-R forecast (planning only).** R1 used 7.9 GPU-h (forecast 3-5). R3 dev: about 1.2 (prep) + 2.9 + about 4.3 (b16) + evaluations = 9-10 GPU-h, against 4-6 in 14.5. R3
  confirmation (25-35 GPU-h in 14.5): about 6 blocks x 3-5 GPU-h + evaluations = 20-35 GPU-h; gate DR records the
  measured value.
- **Next decision points.** Gate DR when the b16 block and the dev evaluations finish. Then 14.9.3 freezes L*, the
  confirmation configs and at most two R1 relationships (Section 11.3), before any R3 confirmation outcome.
- Wave 2 (9.7, owner confirmation 2026-10-29): no Y3 seeds, no filtered Y3, no Y5 redesign. C1 is **unresolved
  within the stated bounds, not absent**. The S1 contrasts stay exploratory.

### 11.3 R1 policy drift (14.1; exploratory, no claim)
Source: jobs 10976378 and 10978077 (scoring), 10979460 (read-only export of yd rows, features and reference
lengths). Analysis: own scripts `audits/scripts/r1_drift_summary.py` and `r1_problem_analysis.py` (stdlib; loss_K
and the c_b baseline reimplemented, not imported). 180 checkpoint files and 24 references were scored; none is missing.
- **Sampling-scoring check: pass for 24 of 24 references** (signed mean 0.0003-0.0016 nats per token; limit 0.005).
- **Trajectories.** KL(initial || current) per sequence, mean over 256 problems and then over 6 seeds, in nats:

| branch | round 64 | round 128 | round 256 | seed sd at 256 |
|---|---|---|---|---|
| b16 GRPO M | 11.9 | 29.7 | 45.3 | 17.3 |
| b128 GRPO M | 13.1 | 24.3 | 33.2 | 10.2 |
| b16 filtered M | 13.7 | 24.1 | 38.6 | 13.0 |
| b128 filtered M | 9.6 | 15.6 | 28.1 | 7.9 |
| b16 MASK / T3 | 20.0 / 25.6 | 33.4 / 42.8 | 43.9 / 60.0 | 18.3 / 17.8 |
| b128 MASK / T3 | 18.0 / 15.1 | 19.7 / 24.5 | 31.4 / 39.7 | 12.8 / 20.5 |
| b32 / b64 GRPO M | 9.2 / 8.7 | 14.7 / 15.4 | 25.8 / 27.3 | 4.2 / 17.9 |

  The mean drift increases with round in all 10 branch types. At round 256, b16 branches drift more than b128
  branches for M, filtered, MASK and T3.
- **Response length dominates the per-problem associations.** The sequence KL is a sum over tokens. Over problems,
  it rises with the reference response length (Spearman +0.42 to +0.69 by branch type). The per-token diagnostic
  falls with length in 3 of 4 branch types (-0.54 to -0.72; b128 GRPO -0.03). The raw associations with excess
  loss_16 have opposite signs under the two measures in 3 of 4 branch types. They are therefore not interpreted.
- **Given length (partial Spearman; M finals only: 24 branches; the two measures agree):** excess loss_16 -0.10 to
  +0.03 (no consistent association); improvement +0.09 to +0.12 at b16 (5 of 6 seeds positive), about 0 at b128;
  initial competence c_a -0.24 to -0.30 at b128 (GRPO and filtered; 0 of 6 seeds positive), about -0.06 at b16.
- **Branch level.** Over 48 GRPO branches, Spearman(drift at 256, look-1 E) = -0.40. Init confounds this value
  (b16: mean drift 49.7, mean E 0.151; b128: 34.8, 0.316). Within init the values are mixed (-0.83 to +0.09).
  Leave-one-seed-out prediction of E (12 GRPO M branches): RMSE 0.072 without drift, 0.077 with sequence drift,
  0.063 with per-token drift. With 12 branches this is inconclusive.
- **Limits.** 72 summaries (4 branch types x 9 variables x 2 measures) without multiplicity correction. The seed
  intervals describe 6 seeds of one dev-panel sample. These are hypotheses for 14.9.3, not findings.

## 12. Changes
- Update 0 (2026-10-26 ~19:45): design, matrix and forecast. No results.
- Update 1 (2026-10-27 04:16): Y1 and dev results (exploratory); matched pairs; reproduction check; canary;
  accounting and forecast. No yf outcome read.
- 2026-10-27 04:17 (scheduled pass): no new result. Timestamps of update 1 and protocol 10.1.2-10.1.4 corrected
  to the git commit times (they had been written as estimates 15-45 min late).
- Update 2 (2026-10-27 12:15): progress, accounting and forecast only. No new result; no yf outcome read; no
  analysis code changed. Workers w6-w9 ended normally (no failed task).
- Update 3 (2026-10-27 20:00): progress and accounting; stage G added (Section 9; now Section 10). Workers report all yf
  evaluations done; look 1 submitted. New script `scripts/a6_yf_accept.py` (structural check before look 1). No yf outcome read.
- 2026-10-27 20:01: timestamps of update 3 corrected to the git commit time (20:00; had been written as 20:20).
- Update 4 (2026-10-28 04:14): reproduction check result (Section 8); accounting; CPU-queue blocker. No yf
  outcome read; no code changed.
- Update 5 (2026-10-28 10:58): look 1 results (Section 9) with structural acceptance and independent recompute
  (Section 8); accounting. Wave-2 decisions pending.
- Update 6 (2026-10-28 12:14): stage-G validation result and precision decision (Section 10); accounting;
  forecast note. No new A6 result.
- Update 7 (2026-10-28 20:15): S1 table (Section 11.1, exploratory); program S and stage R status; wave-2
  decisions (9.7); accounting. No new GPU result; no project code run for this update.
- Update 8 (2026-10-29 04:15): R3 loss blocker resolved (owner, 14.9.1); revised stage-R forecast; next decision
  point gate DR; C1 wording (unresolved, not absent); accounting. No job ran; no new result.
- Update 9 (2026-10-29 12:15): stage-G calibration and dev progress (Section 10); S2 build, gate S2 and S3 dev
  status (11.2); R1 drift results (11.3, exploratory); first R3 measurements and revised stage-R forecast;
  accounting. No confirmation outcome read.
