# Amendment A6 — mechanism follow-up (pre-registration, 2026-10-26)

Purpose: define the follow-up study that owner amendment A6 authorizes (PROJECT.md, 2026-10-26). This file is committed
before any A6 job runs and before any A6 outcome exists. The A5 study (`protocol/AMENDMENT-A5-2026-10-26.md`,
`reports/extension_study.md`) stays complete, and its stop rule stays valid for A5. A6 is a new study. Its report is
`reports/mechanism_followup.md`.

Each stage is pre-registered here before its outcomes exist. Later dated sections (Section 10) may add hypotheses or
stages. They may not change a test after its outcomes exist.

## 1. Established finding and goal

Established (A5, X5 + X7): on fresh 3-input problems that the initial adapter solved in 1-11 of 24 selection attempts,
the standardized increase in loss_16 = (1 - p)^16 after 256 online rounds is larger from the step-128 BC checkpoint
than from the step-16 checkpoint. The difference is +0.203 (X5) and +0.200 (X7), and it is positive in 12 of 12
(method, seed) pairs over seeds 11-66.

Goal of A6: distinguish explanations of this effect. A6 does not replicate it again. The candidate explanations are:

| id | explanation | stage that tests it |
|---|---|---|
| R1 | Measurement: truncation at the 256-token cap or output-format changes hide solutions that are still likely | Y2 |
| R2 | Comparison artifact: the effect is a larger change from a higher baseline, not a worse final state; or matched observed counts hide different true success probabilities | Y1 |
| R3 | Cross-difficulty interference: updates on 5- and 10-input problems damage 3-input solutions, more so from step 128 | Y3 |
| R4 | Same-difficulty sharpening: 3-input training itself concentrates probability on fewer solutions, more so from step 128 | Y3 |
| R5 | BC exposure as such, beyond the initial success distribution that it produces | Y4 |
| R6 | Initial success distribution (competence), not BC exposure | Y4 |
| R7 | Objective: filtered supervision reinforces successful responses differently from GRPO | method interaction (Section 7) |

The explanations are not exclusive. A stage can support several of them.

## 2. Resources (owner A6; binding)
- GPUs only on `preempt` / `preempt_qos`. At most 5 BC2RL GPU slots, counting running and pending allocations and array
  concurrency, subject to any tighter authorization. Run `tools/bc2rl_guard.py submit` before every submission. Do not
  use general GPUs, `ae-topup` or GPU standby. Do not interfere with other projects. CPU-only work: one job at a time on
  `preempt_cpu_qos`, <= 4 CPUs.
- No cumulative GPU-hour cap and no deadline. Account for every allocated GPU-hour (sacct). Update the forecast for
  each stage.
- Scientific comparisons use fixed BC steps, online rounds, prompts per round and attempt counts. They never use
  equal GPU-hours.

## 3. Data and checkpoints

### 3.1 New panels (generator v1, `bc2rl xdata`, data name `a6`)
- Frozen v1 generator configuration (`configs/data_v1.yaml`), new seed **20261027**. A canonical key that occurs in
  any v1 split, the format example, the mechanism splits or the X5 panels (xc3, xc5) is rejected and counted.
- **Development panels** (exploratory diagnosis and matching decisions only): `yd3` = 384 problems with 3 inputs,
  `yd5` = 96 problems with 5 inputs.
- **Confirmation panels** (pre-registered tests only): `yf3` = 768 problems with 3 inputs (primary), `yf5` = 128
  problems with 5 inputs (manipulation checks and whole-panel description).
- No training step, matching decision or hypothesis selection uses yf. The yf outcomes are analysed once per look
  (Section 8), after the tests of that look are frozen in this file.

### 3.2 Checkpoints
- BC steps 16 (light), 32, 64, 128 (extended) of the broad BC paths of seeds 11, 22, 33, 44, 55, 66. Steps 16, 64,
  128 exist (`runs/bc2rl-s1-bc-broad-s<seed>/adapters/`). Step 32 does not exist.
- **Step 32:** the BC path of each seed runs again with the frozen configuration (128 steps; checkpoints 16, 32, 64,
  128; run `bc2rl-y4-bc-path-s<seed>`). The data order is a function of the seed only, and BC uses no random draws.
  Reproduction check: the sha256 of the re-run step-16, step-64 and step-128 adapters against the originals. If they
  differ (GPU numerics), the report gives the relative L2 parameter distance of each pair and the dev pass@1 of both.
  The step-32 adapter always comes from the re-run. Steps 16, 64 and 128 always use the original adapters.
- Existing online endpoints: the 24 round-256 adapters ({16, 128} x {GRPO, filtered} x 6 seeds), arm **M**.

### 3.3 Evaluation protocol (`bc2rl xeval`, as X5 with one addition)
- Prompt, decoding (temperature 1, top-p 1, no top-k, repetition penalty 1), verifier v1 and sampling-seed rule as X5.
- Initial adapters: **48 attempts** per problem; attempts 0-23 = selection (c_a), 24-47 = baseline (c_b).
  Final adapters: **32 attempts**.
- **Two caps from one sample.** Each attempt is sampled with the frozen cap of 256 new tokens. An attempt that reaches
  256 tokens without a stop token continues from its own 256-token prefix up to **1,024** tokens, with the same
  decoding settings. The cap-256 outcome of an attempt is its outcome at 256 tokens (truncated = failure). The
  cap-1,024 outcome is its outcome after continuation. An attempt that stopped before 256 tokens has the same outcome
  at both caps. Reason: autoregressive sampling makes the first 256 tokens of a 1,024-cap sample a 256-cap sample, so
  this gives the two caps on the same attempts (exact pairing). Other decoding settings do not change. The training
  cap stays 256.
- Per attempt, one failure code at each cap: S success; T truncated (no stop token at the cap); F format (malformed
  line or no operation); I illegal operand (an operand not available); A arithmetic error; P incomplete (fewer than
  n - 1 valid operations, no error); W wrong target (n - 1 valid operations, wrong final value). These are the
  verifier's categories (`task.verify`); F joins `malformed` and `no_ops`. A truncated attempt is a failure at that
  cap. It is never excluded.
- Lenient re-parse (diagnostic only, never a reward): every substring of the form `a op b = c` in the text, in order,
  checked as a chain. It shows whether a correct chain is present inside a strict-format failure.

### 3.4 Common estimator (as X5)
For a problem with c successes in n attempts: loss_K(n, c) = C(n - c, K) / C(n, K), unbiased for (1 - p)^K. For an
initial adapter i and a final adapter f trained from i: excess e = loss_K(32, c_f) - loss_K(24, c_b). Problems are
binned on c_a (cap 256) into B1 = 1-2, B2 = 3-5, B3 = 6-11 of 24. The branch statistic is E = sum_b w_b mean_b(e) /
sum_b w_b over the usable bins. A bin is usable in a comparison only when every initial adapter in the comparison has
>= 10 problems in it. The weights w_b are the pooled bin shares over **all 24 initial adapters** on yf3 (selection
counts only). K = 16 unless stated otherwise. Uncertainty: (a) a paired problem bootstrap, with 10,000 resamples of
yf3 problems shared by all adapters and bins fixed on the observed selection counts; (b) a seed-level Student-t 95%
interval (df = 5) over the 6 seed values. **A claim of a nonzero contrast requires both intervals to exclude 0.** The
problem interval is conditional on the trained adapters. The seed interval treats the 6 seeds as independent training
runs evaluated on one shared panel.

## 4. Stage Y1 — interpretation audit (existing X5/X7 data; exploratory; no GPU)
Data: the X5/X7 evaluations of xc3 and xc5 (`$PROJ_DATA/evals/ext/x5/`; seeds 11-66). Script `scripts/y1_audit.py`.
These panels were used for the A5 confirmations, so every Y1 result is labelled exploratory or audit.
1. **Change versus final state.** Per pair, split Delta = (F_ext - F_light) - (B_ext - B_light), where F and B are the
   standardized final and baseline loss_16 on the subgroup (same weights and bins). Report Delta_F, Delta_B and
   Delta at K = 1 and K = 16 with paired intervals. Report the whole-panel final pass@1 and pass@16 of extended minus
   light finals (xc3, xc5), per pair.
2. **Selection and weighting.** Delta under 4 weight vectors (pooled; light-only; extended-only; equal), per bin, and
   with each bin left out.
3. **Seed dependence.** The 12 pair values; Delta_s for the 6 seeds with the t-interval; a two-level bootstrap
   (seeds, then problems).
4. **Observed counts are not true probabilities.** Per initial adapter, a nonparametric maximum-likelihood estimate of
   the distribution of p over problems (EM on a 101-point grid, 48-attempt counts). From it, E[p | c_a bin] for each
   adapter, compared with the observed baseline rate (c_b / 24). This gives the size of the true-probability
   mismatch inside the bins. It is a model-based description, not a correction of Delta.
5. **Method interaction (exploratory here; confirmatory on yf, Section 7).** I_s = Delta(filtered, s) - Delta(GRPO, s).
6. **Failure decomposition at cap 256** (xc3 records): category shares of the excess failure rate, per init and
   method (Section 5.2 method).
7. **Structure and strategy (exploratory).** Per problem: number of distinct valid solution chains (exact enumeration),
   whether every solution needs subtraction or multiplication, target size, reference structure. Per success: the
   solution form (operation order and operators). Association of these features with the excess loss, by init. Results
   can only generate hypotheses. A hypothesis is tested on yf only when Section 10 freezes it before look 1.
8. **Report corrections.** The audit lists every statement in `reports/extension_study.md` that exceeds this evidence.
   Corrections go to a dated corrections section of that report. The A5 protocol file is not changed.

## 5. Stage Y2 — truncation and failure types

### 5.1 Development diagnosis (yd; exploratory)
Adapters: the 12 initial adapters at steps 16 and 128 (48 attempts; yd3 and yd5) and the 24 existing finals (32
attempts; yd3). Per adapter and cap: success, truncation, failure-code rates; length distribution; the outcome at
1,024 of each attempt truncated at 256; lenient-parse successes among strict failures. On the yd3 subgroup: the
Delta statistic at caps 256 and 1,024 and the failure-code decomposition (5.2). Script `scripts/y2_dev.py`.

### 5.2 Confirmation (yf, look 1; pre-registered)
Adapters: the 12 initial adapters at steps 16 and 128 and the 24 M finals on yf3. Pairs: 12 (method x seed).
- **Primary Y2-a (truncation).** Delta_1024: the X5 statistic with baseline and final outcomes at cap 1,024 and the
  subgroup fixed by the cap-256 selection counts. Also Delta_256 and the paired difference Delta_256 - Delta_1024.
  Classification:
  - "not explained by truncation": Delta_1024 lower limits > 0 (both intervals) and Delta_1024 >= 0.5 x Delta_256;
  - "largely explained by truncation": an interval of Delta_1024 includes 0, and both intervals of
    Delta_256 - Delta_1024 exclude 0;
  - otherwise "partly explained", with the values.
- **Y2-b (failure types).** At K = 1, the failure rate is the sum of the code rates, so the excess decomposes exactly:
  e_1 = sum_k (r_k,final - r_k,baseline) over codes k. D_k = E_k(ext) - E_k(light) per pair, and sum_k D_k =
  Delta(K = 1). Report D_k, its two intervals and its share of Delta(K = 1), at caps 256 and 1,024. A code "accounts for
  part of the difference" if both intervals of D_k exclude 0 with D_k > 0. A code with D_k < 0 (both intervals)
  works against the difference.

## 6. Stage Y3 — does training on harder problems contribute to 3-input losses? (GRPO first)

### 6.1 Arms (from the same initial checkpoints: steps 16 and 128, seeds 11-66)
| arm | prompt stream | gradient from | 3-input prompts | total rollouts | optimizer steps |
|---|---|---|---|---|---|
| M (existing) | frozen mixed stream (32 prompts/round from the 4,096-prompt pool) | all completions | 2,732 | 32,768 | <= 256 |
| MASK | identical to M (same stream, same sampling seeds; all 32 prompts are sampled) | 3-input completions only | 2,732 | 32,768 | <= 256 |
| T3 | 32 prompts/round from the 1,366 3-input pool prompts only | all (3-input) completions | 8,192 | 32,768 | <= 256 |
| T3E | T3 at round 85 (saved during the T3 run) | — | 2,720 | 10,880 | <= 85 |

- 8,192 stream slots are exactly 2 passes over the 4,096-prompt pool, so every M stream holds exactly 2 x 1,366 =
  2,732 3-input prompts. T3E at round 85 holds 2,720 (99.6%).
- MASK keeps the frozen GRPO loss and its normalization (sum over used tokens divided by 128 x 256). Only the terms of
  5- and 10-input completions are removed. A round in which no 3-input completion has a nonzero advantage takes no
  optimizer step (it is counted as skipped).
- All other settings are frozen (LR 1e-5, AdamW, clip 0.2, clipping norm 1, G = 4, cap 256, 256 rounds).
- 24 new branches (MASK, T3 x 2 inits x 6 seeds), 36 new final adapters (with T3E). Runs `bc2rl-y3-<init>-grpo-
  <arm>-s<seed>`. The yf evaluation of each final runs after its branch.

### 6.2 What each comparison isolates
- **M vs MASK (primary cross-difficulty contrast).** Same 3-input prompts at the same rounds with the same loss
  scale. The only change is the removal of the 5- and 10-input gradient terms. A difference is caused by the
  harder-problem updates. These include their indirect effects through the changed trajectory and through the AdamW
  normalization. The report gives adapter displacement norms at rounds 64, 128 and 256 for both arms.
- **MASK vs T3.** Both arms update on 3-input problems only. T3 has 3 times the 3-input exposure and no harder
  rollouts. The contrast measures more same-difficulty training.
- **M vs T3 (equal total rollouts).** The usual "3-input-only training" comparison. It mixes the two effects above,
  so it does not isolate either.
- **MASK vs T3E** (secondary). Equal 3-input exposure. T3E has no harder rollouts and fewer optimizer steps.
- Manipulation checks: yf5 pass@1 change (final - initial) for each arm; in the training records, the share of
  gradient-carrying completions by input count (M has 5-input terms; MASK and T3 have none).

### 6.3 Statistics (yf3, look 1)
E(i, a, s) is the branch statistic of Section 3.4 for initial step i in {16, 128}, arm a and seed s. Delta_a(s) =
E(128, a, s) - E(16, a, s).
- **C1 (primary): cross-difficulty share of the init difference:** mean_s [Delta_M(s) - Delta_MASK(s)].
- **C2: the init difference without harder-problem updates:** mean_s Delta_MASK(s).
- **C3: cross-difficulty effect on loss itself:** mean_s [E(i, M, s) - E(i, MASK, s)], for each i.
- **C4: extra same-difficulty exposure:** mean_s [E(i, T3, s) - E(i, MASK, s)] for each i, and mean_s Delta_T3(s).
- Secondary: MASK vs T3E; all contrasts at cap 1,024; K = 1, 4, 24.

### 6.4 Pre-registered predictions and interpretation rules
| outcome | R3 (cross-difficulty) predicts | R4 (same-difficulty) predicts |
|---|---|---|
| C1 | > 0, near Delta_M | near 0 |
| C2 | near 0 | > 0, near Delta_M |
| C3 | > 0 for both inits | near 0 |
| C4 | not specified | >= 0 (more exposure, equal or more loss) |
- "Harder-problem updates contribute to the init difference" if C1 > 0 (both intervals).
- "The init difference arises without harder-problem updates" if C2 > 0 (both intervals).
- Both statements can hold. If neither holds, the result is "undetermined at this precision", and the report gives
  the smallest effect that the intervals exclude.
- Detectable size (planning): the seed SD of Delta_s in A5 was about 0.06, so the seed-level t-interval of a
  difference of two Deltas has a half-width of about 0.08. Effects smaller than about half of Delta_M(GRPO) = 0.16
  may stay undetermined. The report states this.

### 6.5 Extension to filtered supervision (conditional; decided after look 1)
The same MASK and T3 arms with filtered supervision run only if (a) C1 or C2 is decided for GRPO, and (b) the method
interaction (Section 7) or the per-method A5 values make objective dependence a live question. Its pre-registration
(including the MASK normalization for filtered supervision) is written in Section 10 before it starts.

## 7. Method interaction (R7; confirmatory on yf, look 1)
On the M finals (steps 16 and 128, 6 seeds): I = mean_s [Delta(filtered, s) - Delta(GRPO, s)]. "The objective
changes the init difference" only if both intervals of I exclude 0. Otherwise the report states no method difference
in the init effect, with the interval. Per-method Delta values without this test are descriptive. An exploratory
check of why (share of updates from all-success groups; final strategy concentration) uses training records and yd.

## 8. Stage Y4 — BC exposure and initial competence (GRPO)

### 8.1 Training
Online GRPO, frozen settings, 256 rounds, from steps 32 and 64 of seeds 11-66 (12 branches, runs
`bc2rl-y4-step<N>-grpo-s<seed>`). With the existing M finals from steps 16 and 128, this gives 4 BC exposures x 6 seeds
under a fixed online budget.

### 8.2 Development measurements and matching (yd only; decided before look 1)
All 24 initial adapters are evaluated on yd3 and yd5 (48 attempts, cap 256). Per adapter: pass@1 on yd3 and yd5;
the histogram h of per-problem success counts on yd3 over the classes {0}, {1-4}, {5-12}, {13-24}, {25-40}, {41-48}
of 48; the mean baseline rate (c_b / 24) in each primary bin. Two adapters x, y with different BC steps are
**matched** if all of these hold:
- |pass@1_yd3(x) - pass@1_yd3(y)| <= 0.02;
- |pass@1_yd5(x) - pass@1_yd5(y)| <= 0.02;
- total variation 0.5 sum |h_x - h_y| <= 0.10;
- the mean baseline rate differs by <= 0.03 in each primary bin with >= 10 problems for both.
All matched pairs (same or different seeds) enter. The residual mismatch on every criterion is reported. If no pair
matches, the matched comparison is **not done**, and the report states "overlap inadequate". `scripts/y2_dev.py`
writes the matched pairs. Section 10 records them before look 1.

### 8.3 Statistics (yf3, look 1)
- **Dose-response (primary).** E(step, s) for steps 16, 32, 64, 128 (arm M, GRPO). Report the seed means and the
  increments I1 = E32 - E16, I2 = E64 - E32, I3 = E128 - E64 with both intervals. The weights are those of Section 3.4.
  A bin is usable only if all 4 initial adapters of the seed have >= 10 problems in it. Shape rule:
  - "gradual" if all three increments are > 0 (point) and at least two have both lower limits > 0;
  - "threshold" if one increment is > 0 (both intervals) and >= 70% of E128 - E16, and the other two have
    intervals that include 0;
  - "non-monotonic" if any increment is < 0 (both intervals);
  - otherwise "undetermined".
- **Exposure beyond competence (R5 vs R6).** For each matched pair (x lower step, y higher step):
  D = E(y) - E(x) on yf3, with the problem-bootstrap interval of the mean over pairs. "BC exposure predicts loss beyond
  the measured initial distribution" if the mean D > 0 with an interval that excludes 0 **and** all pairs have
  D > 0. If D is near 0 (interval within +-0.05), then the matched initial distribution explains the step effect for
  these pairs. A seed-level interval is not available when pairs share adapters. The report states this.
- **Secondary.** A regression over the 24 checkpoints of E on log2(step) and the yd3 pass@1, with the correlation of
  the two predictors. It is reported as descriptive if |r| > 0.8.

## 9. Stage Y5 — E2 redesign (conditional; not launched with wave 1)
E2 runs only if it can resolve a question that remains after look 1. Example: whether single matched updates from the
identified training composition (Y3) predict later behavioural loss. If it runs, its pre-registration in Section 10
must contain: new development data; an error criterion |actual - predicted| <= a + r |predicted| with an absolute term
a that handles near-zero predictions; validation of numerical precision (repeat noise above zero), curvature, actual
optimizer displacement and an independent held-out effect; the gate frozen before confirmation. A failed validity
check is not evidence against gradient interference. A reference-trace likelihood result alone is not evidence of
preserved verified reasoning. Otherwise Y5 is not run, and the report gives the reason.

## 10. Looks, stopping rule and dated additions
- **Look 1** (`scripts/y_look1.py`, one CPU job): it runs when all wave-1 adapters (Sections 5.2, 6, 7, 8) have yf
  evaluations. It computes every test above and every hypothesis frozen in Section 10 before the job starts. It refuses
  an incomplete set. No yf outcome is read before that job.
- **Wave 2** (filtered Y3, Y5): decided after look 1 under 6.5 and 9. Each new stage is pre-registered in Section 10
  and analysed in its own look.
- **Stop rule.** Stop expanding when the leading explanations are distinguished, or when the remaining uncertainty is
  characterized and no feasible experiment would distinguish the remaining explanations. A negative or
  measurement-limited result is a valid end.
- Exclusions: none. Truncated, malformed and illegal outputs are failures. A branch that fails deterministically is
  reported, and it is not re-run with changed settings. A preemption resumes from the last checkpoint.

### 10.1 Dated additions

**10.1.1 (2026-10-26 ~21:00 EDT; after Y1 on xc3/xc5, job 10947200; before any yd or yf outcome was read).**
Y1 (exploratory, xc3) generated the hypotheses below. They are tested at look 1 on yf3 with the M finals (steps 16 and
128, both methods, seeds 11-66). The subgroup is the A5 one (1 <= c_a <= 11, cap 256). Seed-level unit = the mean over
the 2 methods. A claim needs both the problem bootstrap and the seed-level t-interval to exclude 0, unless stated.
- **H-W1 (whole panel, R2).** The mean over the 12 pairs of whole-panel final pass@16 (extended - light) on yf3 is
  > 0. Same for pass@1. (Y1: 12/12 pairs > 0 on xc3.) If it holds, a larger subgroup loss does not imply a worse
  whole-panel final state.
- **H-F1 (failure types).** E_I(step 16) < 0: online training from step 16 removes illegal-operand failures on its
  subgroup (K = 1 code rate, standardized as Section 3.4). Also reported: E_I(step 128) and E_W for both steps.
- **H-S1 (structure).** For each step, the subgroup mean excess loss_16 (not standardized) on problems whose every
  valid solution uses multiplication, minus that on the other subgroup problems, is > 0. Features come from exact
  enumeration (`yanalysis.solution_forms`).
- **H-S2 (structure x init).** Delta on the needs-multiplication problems minus Delta on the other problems
  (Delta = extended - light subgroup mean excess, not standardized). Two-sided; reported with both intervals.
- **H-S3 (strategy).** Within each branch, the Pearson correlation over subgroup problems between the excess loss_16
  and the drop (initial - final) in the policy's preference for the problem's best solution form is > 0. The claim
  needs the seed-level t-interval only (the correlation has no problem-bootstrap interval here); it is made per step.

**10.1.2 (2026-10-27 ~03:10 EDT; after the dev job 10947639 on yd only; before any yf outcome was read).**
Matched pairs for Section 8.3 (from `y4_matching.json`, Section 8.2 rule, 24 initial adapters, 216 cross-step pairs):

| lower step | higher step | d pass@1 yd3 | d pass@1 yd5 | TV | max d bin baseline |
|---|---|---:|---:|---:|---:|
| b16-s44 | b32-s11 | 0.0134 | 0.0028 | 0.086 | 0.020 |
| b64-s33 | b128-s66 | 0.0109 | 0.0039 | 0.078 | 0.010 |
| b64-s44 | b128-s66 | 0.0012 | 0.0028 | 0.089 | 0.027 |

- No pair spans steps 16 and 128. The matched comparison therefore tests one-step-of-doubling contrasts (16 vs 32,
  64 vs 128) only. It does not test the full 16 vs 128 contrast.
- Two pairs share b128-s66. The problem bootstrap resamples yf3 problems jointly for all pairs; no seed-level interval
  is computed (Section 8.3). The report states the residual mismatch above for each pair.
- D uses the GRPO finals of each adapter (arm M for steps 16 and 128; Y4 branches for steps 32 and 64).
- Y2 on yd (exploratory): no attempt truncated at 256 tokens succeeded after continuation to 1,024 tokens, in any of
  the 48 yd adapters. Delta at cap 1,024 equals Delta at cap 256. The pre-registered Y2-a test on yf is unchanged.
- Step-32 adapters (re-run path) had lower truncation on yd3 (<= 0.0002 per adapter) than steps 16 and 64. The
  Section 3.2 reproduction check runs before look 1 (job `bc2rl-a6-repro-main-s0`); the report gives its result.

**10.1.3 (2026-10-27 ~04:20 EDT; Section 3.2 reproduction result, job 10949310).** The re-run BC adapters are not
bit-identical to the originals (0/18 sha256 equal). Relative L2 distance re-run vs original: step 16 0.003-0.006,
step 64 0.011-0.017, step 128 0.018-0.028. The re-run step-32 adapter is at 0.032-0.033 from the original step 16 and
0.041-0.042 from the original step 64. As Section 3.2 requires, the re-run step-16/64/128 adapters get yd evaluations
(tags `rr<step>-s<seed>`, 6 tasks appended to the worklist). They enter the report only as the reproduction check, not
in any test. The analysis plan is unchanged: step 32 comes from the re-run; steps 16, 64, 128 from the originals.

**10.1.4 (2026-10-27 ~05:00 EDT; after an independent statistical review of `scripts/y_look1.py`; before any yf
outcome was read).** Readings and handling rules fixed for look 1:
- **H-S3 claim** = the seed-level t-interval lower limit > 0 (10.1.1 says "t-interval only"; the earlier code could
  never set the claim). A branch whose correlation is undefined (a constant vector) is dropped; the count of branches
  used is reported. The "best form" drop is max_f pref_initial(f) - max_f pref_final(f) over the problem's valid
  forms, as in the Y1 analysis that generated H-S3. Successes whose form cannot be parsed stay in the denominator.
- **Empty usable set.** A seed (Y3, H-F1, method interaction, Y4) or a matched pair with no usable bin is excluded from
  that comparison and listed in `look1.json` (`excluded_units`).
- **Y2-a.** "An interval includes 0" = at least one of the two intervals contains 0. "Largely explained by
  truncation" also needs Delta_256 - Delta_1024 > 0 (both intervals).
- **Y4 threshold.** "The other two have intervals that include 0" = both intervals of each of the other two
  increments contain 0.
- **Y2-b flags** use the sign rule of Section 3.4 (both intervals on the same side of 0).
- **Y3 secondary scope.** K = 1, 4, 24 at cap 256; cap 1,024 at K = 16. The smallest excluded effect is read from the
  intervals in the report.
- **Manipulation check (6.2).** Shares of gradient-carrying completions by input count for M, MASK and T3; branches
  without a `used_by_inputs` log are listed. Displacement norms are a diagnostic; their failure cannot stop look 1.
- **Input validation before any statistic:** attempt counts (48 / 32), cap-1,024 fields and code strings for every
  tag, yf3 records for the H-S3 tags, the data config and the matching file. Results are written after each section.

## 11. Forecast (from A5 measurements)
Throughput: evaluation 25.7 completions/s averaged over X5 (645,120 completions in 6.97 GPU-h); online GRPO
0.6-1.3 GPU-h per branch (X7); BC about 6 min per seed for 128 steps (A100).
| item | work | GPU-h |
|---|---|---:|
| BC re-runs | 6 seeds x 128 steps | 0.6 |
| yd evaluation | 24 initial x 48 x 480 + 24 finals x 32 x 384 = 847,872 completions | 9-10 |
| Y3 + Y4 training | 36 GRPO branches | 30-40 |
| yf evaluation | 24 initial x 48 x 896 + 72 finals x 32 x 896 = 3,096,576 completions | 33-37 |
| continuation to 1,024 tokens | truncated attempts only (about 2-10%) | 2-5 |
| total wave 1 | | 75-93 |
With 5 slots and about 10% preemption rework: about 17-21 h of wall time after the workers start, plus queue waits.
Wave 2 (conditional): filtered Y3 about 35 GPU-h.
