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

**10.1.2 (2026-10-27 02:55 EDT, commit c3a4311; after the dev job 10947639 on yd only; before any yf outcome was read).**
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

**10.1.3 (2026-10-27 04:11 EDT, commit d1795e0; Section 3.2 reproduction result, job 10949310).** The re-run BC adapters are not
bit-identical to the originals (0/18 sha256 equal). Relative L2 distance re-run vs original: step 16 0.003-0.006,
step 64 0.011-0.017, step 128 0.018-0.028. The re-run step-32 adapter is at 0.032-0.033 from the original step 16 and
0.041-0.042 from the original step 64. As Section 3.2 requires, the re-run step-16/64/128 adapters get yd evaluations
(tags `rr<step>-s<seed>`, 6 tasks appended to the worklist). They enter the report only as the reproduction check, not
in any test. The analysis plan is unchanged: step 32 comes from the re-run; steps 16, 64, 128 from the originals.

**10.1.4 (2026-10-27 04:16 EDT, commit b76839f; after an independent statistical review of `scripts/y_look1.py`; before any yf
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

## 12. Stage G — sequence-level weighting and clipping (GSPO); owner addition 2026-10-27

Pre-registered 2026-10-27, before any stage-G job. Owner instruction: add a bounded GSPO experiment without displacing
the mechanism work (A6 wave 1 and look 1 keep priority on the 5 BC2RL GPU slots).

**Question (owner, verbatim).** Does sequence-level weighting and clipping change coverage loss or its dependence on BC
initialization, beyond effects from response-length normalization and rollout reuse?

### 12.1 Objectives (`bc2rl/losses.py`; config key `online.objective`)
Advantages as GRPO: A_i = (r_i - mean_g) / (std_g + 1e-6), population std, G = 4. Q = 128 completions per round.
| id | objective | loss |
|---|---|---|
| `grpo_const` | reference (the A5 / A6 GRPO; unchanged code path) | -(1 / (Q Lmax)) sum_i sum_t min(r_it A_i, clip(r_it, 1-e, 1+e) A_i), Lmax = 256 |
| `grpo_len` | token ratio, response-length normalization | -(1 / Q) sum_i (1 / abs(y_i)) sum_t min(r_it A_i, clip(r_it, 1-e_tok, 1+e_tok) A_i) |
| `gspo` | sequence ratio (GSPO, arXiv 2507.18071, Eq. 5-7) | -(1 / Q) sum_i min(s_i A_i, clip(s_i, 1-e_seq, 1+e_seq) A_i), s_i = exp((1 / abs(y_i)) sum_t log(p(y_it) / b(y_it))) |
- abs(y_i) = completion tokens including the stop token; a truncated attempt has abs(y_i) = 256. Padding after the stop
  token has no loss and no gradient. Rows with A_i = 0 contribute 0; the denominator stays Q.
- **Behaviour log-probs b** are the detached log-probs of the first gradient pass of the round. They stay fixed for
  all K steps of the round. The ratio of step 1 is therefore exactly 1.
- **K full-batch optimizer steps per rollout batch** (`online.updates_per_batch`): each step accumulates the gradient
  over the whole batch (same micro-batch layout each step), clips the gradient norm at 1.0, and makes one AdamW step.
- **Identity at K = 1.** With ratio 1, `grpo_len` and `gspo` have the same gradient,
  -(1/Q) sum_i (A_i / abs(y_i)) sum_t grad log p(y_it). So at K = 1 they are the same algorithm. Arm L1 represents both.
  The difference between `gspo` and `grpo_len` can act only at steps 2..K.

### 12.2 Arms (per seed and initial adapter: light = BC step 16, extended = BC step 128)
| arm | objective | K | clip band |
|---|---|---:|---|
| C1 | `grpo_const` | 1 | 0.2 (inactive: ratio 1) |
| L1 | `grpo_len` | 1 | e_tok (inactive: ratio 1) |
| L4 | `grpo_len` | 4 | e_tok (12.4) |
| S4 | `gspo` | 4 | e_seq (12.4) |
Everything else is the frozen M-arm configuration (`configs/main/on-<init>-grpo-s<seed>.yaml`,
`configs/ext/x7/...` for seeds 44-66): mixed 3/5/10-input prompt stream, 32 prompts x 4 samples, 256 rounds, cap 256,
temperature 1, verifier v1 reward, LoRA r 16, AdamW lr 1e-5, weight decay 0, fresh optimizer state, same stream seed and
sampling seeds for all four arms (round 1 rollouts are identical across arms). Training-pass precision: 12.3 (V4),
the same for all four arms. KL to the initial adapter every 8 rounds (`online.kl_every`).
**Contrasts.** Normalization: L1 - C1. Rollout reuse: L4 - L1. **Sequence-level weighting and clipping: S4 - L4**
(same normalization, same reuse). BC dependence: each contrast for extended minus light (interaction).
A difference between L1 and C1 is a normalization effect. It is never attributed to sequence-level clipping.
The existing A5 / A6 M-arm GRPO runs stay as a separate historical reference (C1 re-runs them with the diagnostics).

### 12.3 Validation (before any probe or dev run)
- V1 masks and stop tokens; V2 clipping branches (A > 0 / A < 0, inside / outside, asymmetric bounds); V3-unit
  first-update agreement and closed form; V5 fp64 gradient checks (finite differences; GSPO Eq. 10); V6 the reference
  path is bit-identical with the diagnostics on. CPU tests `tests/test_gspo.py`, `tests/test_a6.py` (tiny model:
  K = 4 runs, step-1 ratio exactly 1, resume inside a K = 4 run bit-identical, K > 1 refused for `grpo_const`).
- GPU job `scripts/g_validate.py` (dev seed 11, both initial adapters, 2 real rollout batches each):
  - V3: accumulated gradients of `grpo_len` and `gspo` at the first update agree (FP32: relative L2 <= 1e-4 and
    cosine >= 0.99999).
  - V7: two BF16 passes with the same layout give identical log-probs (reported).
  - V4 precision: after 1-3 GSPO steps, the token and sequence log-ratios in BF16 and FP32 (same layout).
    noise = BF16 - FP32. **Rule:** use BF16 training passes if, for every adapter and batch after step 1,
    median abs(noise) <= 0.1 x median abs(FP32 log-ratio), at both the sequence and the token level. Otherwise use
    FP32 training passes (frozen base weights cast to FP32, TF32 off; sampling stays BF16) for all four arms.
- A failed V1-V3, V5 or V6 stops stage G until the defect is fixed and the validation is run again.

### 12.4 Clip calibration (dev seed 11 only; before the dev arms)
- Four probe runs: {light, extended} x {`grpo_len`, `gspo`}, K = 4, **no clipping**, 32 rounds, ratio dumps.
- e_tok = 85th percentile of abs(r_it - 1) over the tokens of rows with A_i != 0, steps 2-4, all probe rounds, both
  initial adapters (`grpo_len` probes). e_seq = 85th percentile of abs(s_i - 1) over those rows (`gspo` probes).
  Symmetric band, 2 significant digits. Reason: each band leaves 15% of its own units outside in the unclipped probe,
  so the two constraints are comparable by unit share; equal numbers would not be (owner instruction). This rule is a
  choice. The report also gives the probe fractions outside the paper values (token 0.2 / 0.27; sequence 3e-4 / 4e-4).

### 12.5 Development (seed 11; exploratory)
- 8 runs: {C1, L1, L4, S4} x {light, extended}. Final adapters (round 256) on yd3 / yd5, 32 attempts, cap 256 with
  continuation to 1,024 (Section 3.3). Initial-adapter counts: the existing yd-init-s11 evaluation.
- **Gate D (validity and information; decided on training diagnostics only, never on a coverage outcome):**
  - D1 validity: V1-V7 as above; all 8 runs complete 256 rounds with finite loss; step-1 ratio exactly 1 in L4 and S4.
  - D2 exercise: in S4 the median over trained rounds of the step-4 median abs(log s_i) >= 3e-4 (the paper's lower
    GSPO clip range) and >= 3 x the V4 noise floor of the chosen precision; in L4 the same for the token median
    abs(log r_it); and in both S4 and L4 the mean clipped fraction (steps 2-4) >= 0.01.
  - If D1 or D2 fails, stage G stops. The report states that the experiment does not exercise the distinction (D2) or
    is invalid (D1), with the measured values. No setting is changed to pass the gate.
- If gate D passes, the configuration (precision, e_tok, e_seq) is frozen in a dated addition before confirmation.

### 12.6 Confirmation (seeds 22, 33, 44, 55, 66; prespecified)
- 40 runs: {C1, L1, L4, S4} x {light, extended} x 5 seeds, frozen configuration.
- **Independent panels** (`bc2rl xdata`, data name `g1`, seed 20261028, generator v1; keys in any v1 split, the
  mechanism splits, X5 and A6 panels are rejected): `gf3` = 768 problems with 3 inputs (primary), `gf5` = 128 with
  5 inputs (description). No stage-G decision uses gf. Initial adapters: 48 attempts (24 selection c_a, 24 baseline
  c_b); finals: 32 attempts; caps 256 and 1,024 as Section 3.3.
- Estimator as Section 3.4 on gf3 (K = 16; bins on c_a 1-2 / 3-5 / 6-11; usable-bin rule; weights = pooled bin shares
  of the 10 initial adapters on gf3). E(arm, init, seed) = weighted mean excess loss.
- **Primary tests (look G, one CPU job after all 40 finals and 10 initial adapters have gf evaluations):**
  - H-G1 main effect: mean over the two inits of E(S4) - E(L4).
  - H-G2 interaction: [E(S4, ext) - E(S4, light)] - [E(L4, ext) - E(L4, light)].
  - Intervals: paired problem bootstrap (10,000 resamples of gf3) and seed-level Student-t 95% (df = 4). A claim of a
    nonzero contrast requires both intervals to exclude 0. No multiplicity correction; two primary tests.
- Secondary (same intervals; labelled secondary): L4 - L1 and L1 - C1 (main effects and interactions), the per-init
  contrasts, and E(C1, ext) - E(C1, light) (the A5 effect in this stage).
- Reported for every arm: pass@k and pass^k (k = 1, 4, 16; unbiased from 32 attempts; caps 256 and 1,024), E at
  K = 1, 4, 16, output length, truncation at 256 and outcome after continuation, KL to the initial adapter, token and
  sequence ratio distributions (quantiles per step), clipped fractions (units and tokens), gradient and update norms,
  distance from the initial adapter. Equal-rollout endpoint: round 256 (32,768 completions per run). Optimizer steps
  (K x trained rounds) and measured GPU-seconds are reported separately.
- Interpretation: S4 - L4 is the only contrast that answers the question. If neither H-G1 nor H-G2 is claimed, the
  result is "no detectable effect at this scale", bounded by the intervals. If clipping is rare or ratios stay near 1,
  the report states that the experiment barely exercises the distinction, whatever the coverage outcome.

### 12.7 Resources and forecast
- Preempt only; at most 5 BC2RL GPU slots across all stages, pending jobs included; no general GPUs, no `ae-topup`.
- Forecast (from A6 measurements; K = 4 runs assumed 1.5-3x a K = 1 run): validation 0.3 GPU-h; probes 0.5-1;
  dev 8 runs 9-15 + yd evaluation 1-2; confirmation 40 runs 50-80 + gf evaluation (10 x 48 x 896 + 40 x 32 x 896 =
  1,576,448 completions) 17-22. Total 80-120 GPU-h. Stage-level forecasts are updated in the 8-hourly report.

### 12.8 CTPO addition (owner, 2026-10-27; pre-registered before any stage-G job output was read)
**Question (owner, verbatim).** Does cumulative-prefix correction change coverage loss and its dependence on BC
initialization, compared with token-level GRPO and sequence-level GSPO?

**Formulation (paper arXiv 2605.07331 Sec. 3.3; reference `cum-token-cumprod-la` in horizon-llm/CTPO
`verl/trainer/ppo/core_algos.py`).**
- d_t = clamp(log p(y_t) - log b(y_t), -20, 20); C_t = sum over t' <= t of d_t' (log space, masked).
  log rho_t = sg[C_t] + log p(y_t) - sg[log p(y_t)], clamped at <= 10. The value is the cumulative ratio. The gradient
  flows only through the current token: dL/d log p(y_t) = -A rho_t / abs(y) for an unclipped token. A plain
  derivative of the cumulative product would add prefix-gradient terms; the reference does not do this, and our
  implementation follows the reference (test: `tests/test_gspo.py`, small exact example).
- Position-adaptive band in log space: [-e_lo t^p, e_hi t^p], t = 1-based completion position; reference defaults
  e_lo 0.025, e_hi 0.05, p 0.5. Dual clip for A < 0 at exp(e_c t^p), reference e_c 0.05. Per-token surrogate
  max(-A rho, -A rho_clip), then min with -A exp(e_c t^p) for A < 0 (reference semantics).
- **Theory scope.** Proposition 1 (unbiased policy gradient) assumes a token-level advantage that depends only on the
  prefix (x, y_1..t), deterministic transitions, no clipping and an exact importance weight. Our training uses a
  group-normalized outcome advantage (it depends on tokens after t, sampled from b), clipping and a dual clip. So
  Proposition 1 does not cover our estimator. **We make no unbiasedness claim** for the implemented CTPO; it is a
  practical clipped estimator. Proposition 2 (variance) is a statement about ratios, not about our coverage outcomes.
- **Identity at K = 1:** rho_t = 1 at the first update and the band contains 1, so CTPO has the same gradient as
  `grpo_len` (and GSPO). Arm L1 represents CTPO at K = 1. No CTPO K = 1 run is made (no duplicated run).

**Controlled differences (method-specific; documented).** Same reward, advantages, sampling, prompts, rounds,
optimizer, K and per-sequence length normalization (-(1/Q) sum_i (1/abs(y_i)) sum_t) as L4 / S4. Kept from the
reference because they define CTPO-LA: the log-space position-adaptive band, the asymmetric shape e_hi = 2 e_lo, the
dual clip with e_c = e_hi, and the clamps (+-20, <= 10). Not used: the reference's global token-mean aggregation (the
paper's formula is the per-sequence mean, which also matches L4 / S4), its 16 mini-batch updates per rollout batch
(owner design: K = 1 vs 4 full-batch steps), and its rollout-to-trainer importance correction (our behaviour log-probs
come from the training pass, so there is no sampler mismatch term).

**Arms (added to 12.2).** T4 = `ctpo`, K = 4, position-adaptive band (p = 0.5). T4F = `ctpo`, K = 4, fixed band
(p = 0; ablation of the clipping schedule).
- Contrasts: T4 - L4 (cumulative vs token ratio, both with per-token clipping), T4 - S4 (cumulative vs sequence
  ratio), each with its extended - light interaction. T4 - T4F isolates the clipping schedule from the ratio.

**Calibration (added to 12.4; dev seed 11 only).** Two more unclipped K = 4 probes ({light, extended} x `ctpo`,
32 rounds) dump C_t and t. e_lo is solved so that 15% of the probe tokens (rows with A != 0, steps 2-4, both inits)
fall outside the band with the reference shape (e_hi = 2 e_lo, e_c = e_hi): for p = 0.5 on C_t / sqrt(t) (T4) and for
p = 0 on C_t (T4F). Same unit share as the GRPO and GSPO bands. The report also gives the probe fraction outside
the reference band (0.025 / 0.05, p = 0.5).

**Validation (added to 12.3).** CPU tests: reference transcription parity (values and gradients, p = 0.5 and 0),
first-update closed form, small exact example (clipped tokens, no prefix gradient, naive cumulative-product gradient
differs), dual and lower clip for A < 0, masking and positions, clamps (finite at log-ratio 100), infinite band, fp64
gradcheck; tiny-model online runs (CTPO K = 1 equals `grpo_len`; K = 4 probe dumps; fixed band). GPU job V3: CTPO and
`grpo_len` gradients agree at the first update (same threshold). V4: the precision rule also requires
median abs(noise) <= 0.1 x median abs(FP32 value) for the cumulative log-ratio C_t.

**Development (added to 12.5).** Dev runs: C1, L1, L4, S4, T4, T4F x {light, extended} = 12 runs + yd finals. Gate D
for CTPO: D1 as 12.5; D2 for T4: median over trained rounds of the step-4 median abs(C_t) >= 3e-4 and >= 3 x the V4
noise floor of C_t, and mean clipped token fraction (steps 2-4) >= 0.01. If T4 fails D2, CTPO is reported as "not
exercised" and is not confirmed; GSPO confirmation is not affected (and vice versa).
- Diagnostics by token position (buckets 1-8, 9-32, 33-64, 65-128, 129-256): abs(C_t) quantiles for every K = 4 arm;
  CTPO fractions outside and clipped (`pos.csv`).

**Confirmation (added to 12.6).** T4 joins the 40 runs (50 runs: 5 arms x 2 inits x 5 seeds, seeds 22-66, gf panels).
T4F joins only if the dev data show an apparent CTPO benefit: dev point estimate of mean-over-inits
E(T4) - E(L4) < 0 or E(T4) - E(S4) < 0 on yd3 (lower coverage loss). This trigger is fixed now; it uses dev data only.
- Primary tests for the CTPO question (look G, same intervals and claim rule as 12.6): H-G3 T4 - L4 main effect;
  H-G4 its interaction; H-G5 T4 - S4 main effect; H-G6 its interaction. With H-G1 and H-G2 there are 6 primary tests at
  95% each; the familywise error is not controlled and the report says so. If T4F is run: T4 - T4F (main and
  interaction) is secondary and interprets a CTPO benefit (schedule vs ratio). A benefit of T4 is attributed to the
  cumulative ratio only if T4F shows it too.
- Null and untested results: if the K = 4 ratios stay near 1 (D2 values near the thresholds), the report states that
  the importance-weighting differences between GRPO, GSPO and CTPO are largely untested, whatever the coverage outcome.

**Forecast (added to 12.7).** Probes +2 (0.3-0.5 GPU-h); dev +4 runs (5-8) + yd evaluation of 4 more finals (0.5);
confirmation +10 T4 runs (13-20) + 10 x 32 x 896 gf completions (3-4); T4F conditional +10 runs (13-20) + 3-4.
Stage G total with CTPO: about 105-155 GPU-h without T4F, 120-180 with T4F.

### 12.9 Dated addition (2026-10-27 15:32 EDT; after an independent code review; before any stage-G job output)
- Definitions made explicit: the diagnostics of step k (ratio quantiles, clipped fractions) are measured in the
  gradient pass of step k, before its update; so "step 4" is the policy after 3 updates of the round. Gate D2 uses
  this definition for all arms. Per-round KL is measured on the round's rollouts before its update (after r - 1
  rounds); the final adapter's KL is measured once on one extra rollout batch sampled from it (`kl_final`).
- Code fixes before any run: `grpo_len` / `gspo` / `ctpo` refuse a config without explicit `clip_low` and `clip_high`;
  an explicit `clip_c: null` means no dual clip; a non-finite loss or gradient norm stops the run (deterministic
  failure, no retry); a tiny-model test now checks the first-update gradient of all three objectives against the
  closed form, magnitude included. No statistical rule changed.

### 12.10 Dated addition (2026-10-28 11:11 EDT; validation result, job 10967595; before any probe run)
Validation ran on 2 initial adapters (light, extended; seed 11) and 2 real rollout batches each (NVIDIA L40S). One
light batch had no row with A != 0 and was skipped by design, so 3 batches enter.
- **V3 pass.** At the first update (behaviour = current), the gradients of `gspo` and `ctpo` agree with `grpo_len`
  in FP32: relative L2 <= 1e-5 (limit 1e-4), cosine 1.0 (limit 0.99999), for all 3 batches.
- **V7 pass.** Two BF16 no-grad passes with the same layout give identical log-probs (max |diff| = 0).
- **V4: BF16 fails; all stage-G arms train with FP32 passes** (`online.train_precision: fp32`; sampling stays
  BF16). After one GSPO step, the median |BF16 - FP32| sequence log-ratio is 0.0028-0.0039, against a median FP32
  sequence log-ratio of 0.0040-0.0081: noise 48-78% of the signal (limit 10%). Token and cumulative levels fail too.
- Measured scale after one full-batch step (FP32; medians): sequence |log s| 0.004-0.008, token |log r|
  3e-6 to 1.2e-4, cumulative |C_t| 0.06-0.13. These are inputs for the probes, not calibration values (12.4).
- Timing: an FP32 gradient pass takes about 1.8-1.9x the BF16 pass (0.35 vs 0.20 s and 1.08 vs 0.58 s; the first
  FP32 pass, 13.6 s, includes warm-up and is not used). Forecast changes go to the report.
- No statistical rule changed.

### 12.11 Dated addition (2026-10-29 07:18 EDT; clip calibration, job 10978094; before any dev run)
The six unclipped probes (12.4, 12.8; seed 11, 32 rounds, FP32 passes) ran in jobs 10968782 and 10970808. The rule
of 12.4 / 12.8 gives the bands below. No value was chosen by judgement. Source: `configs/g/g_calibration.json`.
- Rounds with a dump: 29-32 per probe. The rounds without a dump had no row with A != 0 (light `grpo_len` rounds
  14, 25, 26; light `gspo` round 26; light `ctpo` round 30), as 12.4 requires.
- **e_tok = 0.011** (raw 0.01088; 308,940 tokens). **e_seq = 0.016** (raw 0.01558; 3,348 sequences).
- **T4 (p = 0.5): e_lo = 0.084, e_hi = e_c = 0.17**; 15.1% of 212,478 probe tokens outside at the rounded values.
- **T4F (p = 0): e_lo = 0.74, e_hi = e_c = 1.5**; 14.9% outside.
- Fractions outside the paper values (reported, not used): token band 0.2 / 0.27: 1.4% (light), 1.3% (extended).
  Sequence band 3e-4 / 4e-4: 95.0% (light), 92.3% (extended). CTPO reference band (0.025 / 0.05, p = 0.5): 45.6%
  (light), 36.3% (extended). So the paper sequence band would clip most sequences in this setting.
- Dev runs (12.5 with 12.8): 12 runs {C1, L1, L4, S4, T4, T4F} x {light, extended}, seed 11, FP32 passes (12.10),
  checkpoints at rounds 64 / 128 / 256 (the frozen light config also saved round 240; dev drops it), KL every 8 rounds.
  Configs: `scripts/make_g_configs.py dev`; worklist `g-dev` (12 runs + yd finals). Gate D and all thresholds are
  unchanged.

## 13. Program S — learning signal, transfer between problem classes, and solution concentration (owner addition 2026-10-28)

Owner addition A6-S (PROJECT.md, 2026-10-28 ~17:15 EDT). Recorded at 17:20 EDT, before any program-S job and before any
program-S outcome exists. It adds to A6 and supersedes no earlier section. Stage G (Section 12) continues unchanged.
This section fixes the questions, the stage order, the design rules and the decision rules. Each stage also gets a
dated addition (13.11) with its exact sizes and thresholds. That addition is committed after the stage's development
work and before any of its confirmation data is generated or read.

### 13.1 Questions and competing explanations
Goal: distinguish three explanations of the coverage loss, then test one intervention that addresses the supported
explanation. A negative or measurement-limited result is a valid end.

| id | explanation | evidence that supports it | stage |
|---|---|---|---|
| Q-SIG | Insufficient learning signal: the vulnerable problems produce few mixed-reward groups, so their solutions get little reinforcement | few informative groups for the vulnerable class; loss independent of other-class updates | S3 (signal counts), S6 |
| Q-TRF | Harmful transfer: updates on one problem class reduce success on another class | masked-control contrast T > 0 (13.4) | S1, S3 |
| Q-CON | Concentration: successful outputs move onto fewer solution forms; forms with little successful training disappear | form concentration at matched success counts; losses on low-support forms | S4, S6 |

The explanations are not exclusive. The intervention stage (S5) tests whether a loss can be reduced. A positive S5
result alone does not prove gradient interference.

### 13.2 Stage S1 — complete the existing cross-difficulty question (no new GPU work)
Data: look-1 yf3 counts (export job 10968369). Estimator, bins, weights and intervals: Section 3.4. Script:
workspace `audits/s1_cross_difficulty.py` (own code; it reproduces the 8 look-1 E points exactly).
- Report separately, for each initialization (step 16, step 128) and for the difference (step 128 - step 16):
  harder = M - MASK (effect of including the harder-problem updates); totalmix = M - T3 (equal total sampling);
  exposure = MASK - T3E (equal 3-input exposure; T3E has fewer optimizer steps); more3 = T3 - MASK.
- A near-zero init interaction (C1) is not evidence that harder-problem updates have no effect. The report gives
  the per-init contrasts and the effects that each interval excludes.
- **Precision plan (set 2026-10-28 17:20; it uses the seed SD only, not the point estimates).** Meaningful size
  delta = 0.25 x Delta_M(K = 16) = 0.039. For a contrast with seed SD s, n_req = the smallest seed count with
  t(0.975, n - 1) s / sqrt(n) <= delta. New paired seeds (new BC paths, then M and MASK from steps 16 and 128) are
  added only if n_req <= 12 for C1, the primary open Y3 contrast. Observed: s(C1) = 0.083, n_req = 20. **Decision:
  no seeds are added to Y3.** The S3 masked controls test transfer directly and at lower cost. The per-init
  contrasts have n_req 8-9; they are reported with their bounds and get no seeds.
- Wave-2 decisions (Section 10): filtered Y3 (6.5) is not run now; it can return only through a dated addition if a
  program-S result makes the objective dependence a live question. Y5 (E2 redesign, Section 9) is not run: the S3
  behavioural masked controls address the transfer question, and the E2 assay had no valid radius (X6).

### 13.3 Stage S2 — a 3-input dataset that separates structure from difficulty
Generator v1 rules, prompt, decoding and verifier v1 do not change. New data name `st1`, new seed 20261028. A
canonical key in any earlier split or panel (v1 splits, format example, mechanism splits, X5, a6, g1) is rejected
and counted.
- **Exact enumeration.** For each problem, enumerate every valid solution under the verifier rules (any operand
  pairing and order; negative, zero and large intermediates are legal). Three levels are counted separately:
  *traces* (ordered operation chains; operand order of + and * collapsed), *forms* (the canonical expression over the
  input values: + and - flattened to a signed multiset of terms, * flattened to a multiset of factors, then sorted
  recursively; so (a+b)-c and (a-c)+b are one form), and *shapes* (a form with the values replaced by placeholders,
  such as `x*y+z`).
- **Labels.** Class A (no multiplication required): at least one form uses only + and -. Class B (multiplication
  required): every form uses *. Also: minimum number of * over forms; number of traces and of forms; whether every
  solution needs a negative intermediate; input maximum, input sum and target; initial success of each starting
  checkpoint, measured on independent attempts (a prescreen, never reused as an evaluation outcome).
- **Development inspection (exploratory; dev candidates only).** Distributions by class of: initial success (light
  and extended, seed 11), number of forms, magnitudes, minimum multiplications and response length. The two classes
  are not assumed to differ only in operator use.
- **Training pools.** Built from dev candidates by coarsened exact matching on: initial-success bin under the light
  and the extended seed-11 adapters, number-of-forms bin and target-magnitude bin. A and B get equal counts per
  matched cell. The dated addition reports the cells without common support and the residual differences. It never
  forces a match in a cell without support.
- **Evaluation panels** (fresh problems, disjoint from all training candidates; dev and confirmation versions are
  separate draws): (1) **representative panel** `sr3` = a generator sample with the natural class mix; (2) **diagnostic
  panel** `sd3` = balanced over class x prescreen-success range. Results on the two panels are reported separately.
  The manifest reports how deduplication, exclusion and balancing change the distribution.
- **Gate S2** (on dev data only): each class has enough unique problems for the S3 pools (the size is fixed in the
  dated addition) and common support in initial success for both inits. If the gate fails, or if 3 inputs give too
  few shapes for S4, a separate 4-input extension is pre-registered (classes from all valid solutions, not the
  generator tree; learnability on dev first). The 3-input study stays a distinct result.

### 13.4 Stage S3 — which classes help or harm each other (GRPO, established configuration)
Initial adapters: light (BC step 16) and extended (step 128) of each seed. The frozen A5/A6 GRPO configuration
applies (32 prompts per round, G = 4, cap 256, 256 rounds, LR 1e-5, AdamW, the constant loss normalization, the
same decoding and reward). Every arm uses the same number of unique training problems and the same passes per problem.

| arm | prompt stream | gradient from |
|---|---|---|
| A | pool A only | all completions |
| B | pool B only | all completions |
| AB | 16 A + 16 B prompts per round (fixed split) | all completions |
| AB-mB | identical stream and sampling seeds to AB | A completions only (B terms masked; budget not reallocated) |
| AB-mA | identical stream and sampling seeds to AB | B completions only |

- Masking keeps the loss normalization of AB, as Y3 MASK. One AdamW state is kept for the run; masked terms give zero
  gradient; no state reset. A round with no unmasked completion with a nonzero advantage takes no optimizer step and
  is counted as skipped. The contrasts are the total effect of including the updates, including their effects through
  the optimizer state and the later policy.
- **Primary contrasts** (diagnostic panel, class subgroup with 1 <= c_a <= 11 of 24, standardized excess E at
  K = 16 as Section 3.4): T_B->A(i) = E_A(AB) - E_A(AB-mB) and T_A->B(i) = E_B(AB) - E_B(AB-mA), for each init i, and
  their init difference. **Transfer table** (descriptive, with intervals): arm in {A, B, AB} x evaluation class in
  {A, B}, for E and pass@1 change.
- Signal records per round and class: all-success groups, all-failure groups, mixed groups, gradient-carrying
  completions and tokens; every 8 rounds the norm of each class's gradient and their cosine (diagnostic). Equal
  sampled prompts do not imply equal learning signal. If a class has almost no informative groups, the report states
  that limit and does not conclude that its structure cannot transfer.
- Development: seed 11, all 5 arms x 2 inits, dev panels. Confirmation: seeds from 22, 33, 44, 55, 66; at least 3.
  The dated addition fixes the seed count with the rule of 13.2 (planning SD = the Y3 C3 seed SD, 0.049) and fixes
  delta for S3. One look; no seeds added after the look.

### 13.5 Stage S4 — structural diversity vs more examples (only if chosen by 13.10)
- **Diversity contrast:** a narrow pool (few shapes) vs a broad pool (more shapes) with equal unique problems, equal
  sampled completions and rounds, approximately equal initial-success distribution, and other difficulty variables
  matched where overlap permits.
- **Breadth vs repetition:** within one fixed structural mixture, 64 vs 256 unique problems (adjusted on dev data if the
  generator cannot supply them) with equal sampled completions.
- Evaluation: fresh instances of the represented shapes; withheld shapes, where the partition permits; the
  representative panel. If diversity and difficulty cannot be separated, the result is reported as the joint
  intervention and is not called a diversity effect.

### 13.6 Stage S5 — a preservation intervention (only if chosen by 13.10)
- The vulnerable class is chosen on development evidence and frozen in a dated addition before confirmation.
- Rehearsal set: independent training problems that the starting policy solves intermittently (prescreen 1-11 of 24);
  verified solutions generated by that starting policy. No evaluation problem, and no selection by confirmation losses.
- Arms: (1) online only; (2) online + rehearsal from the vulnerable class; (3) online + an equal-size broad rehearsal
  set from all intermittently solved training problems. Arms 2 and 3 use the same rehearsal tokens and the same
  coefficient (one coefficient calibrated on dev data, then frozen), and are matched in baseline success and solution
  length where feasible. Online rollout exposure is equal in all arms; the extra training cost is reported.
- Report coverage on the vulnerable class, accuracy and consistency, other classes, and whether rehearsal suppresses
  learning everywhere; an accuracy-coverage plot; a comparison with an earlier online checkpoint of similar accuracy
  (preservation vs slower training).

### 13.7 Stage S6 — solution strategies (diagnostic)
On a diagnostic subset fixed in the S3 dated addition, the full generated texts are kept (xeval records). For correct
initial and final outputs: the forms used, the concentration of successful outputs (form entropy and top-form share),
forms that disappear, and whether losses fall on forms with little successful online training. Comparisons use equal
attempts and adjust for success counts (rarefaction to equal numbers of successes). Fewer observed forms with fewer
successes is not by itself a collapse. S6 can select a later intervention only through a new prospective comparison.

### 13.8 Stage S7 — oracle-assisted search (conditional; frozen policy)
Only if the main stages leave open whether lost solutions stay reachable through intermediate choices. Sample
candidate next operations, check validity and solvability of the resulting state with the exact solver, and compare
unguided and solver-guided continuation at a matched generation budget. Label: oracle-assisted search. It cannot show
why online training changed the policy.

### 13.9 Common rules for program S
- Each new experiment family: one development configuration (seed 11, dev panels) first; then frozen confirmation
  settings; at least 3 paired seeds; seeds added only under a precision plan fixed before confirmation.
- Report for every comparison: pass@1 and the full supported pass@k / pass^k curves; final value and change from
  initialization; whole-panel and class-subgroup results; independent selection (c_a) and baseline (c_b) attempts for
  intermittent subgroups; failure types, truncation, learning-signal counts, optimizer steps and resource use; the
  paired problem interval and the seed interval separately. A claim needs both intervals to exclude 0.
- Subgroups stay fixed across arms that share an initial checkpoint. Equal observed counts are not equal true success
  probabilities.
- Program-S runs reuse existing checkpoints and evaluations only when the protocol matches (same configuration, panel
  and attempt rule). Stage-G runs are not reused in program S.

### 13.10 Order, decisions, stopping and resources
- Order: S1 (now, CPU) -> S2 (data, then GPU prescreen) -> S3 dev -> S3 confirmation -> one of S4 or S5 -> the other
  only if it can still distinguish the remaining explanations -> S7 only if needed. S6 runs on the records of S3 and
  of the chosen stage. The full product of structures, algorithms, BC lengths and controls is not run.
- Choice after S3 (recorded as a dated addition with the reason, before any S4/S5 run): S5 if S3 or S6 identifies a
  vulnerable class or form set; S4 if S3 shows that the loss depends on which classes or shapes are trained and no
  class is clearly vulnerable. If neither, the report states which explanation stays open and why no feasible stage
  separates it.
- Stop: as Section 10. A failed gate leads to diagnosis or another authorized task, never to relaxed validity rules.
- Resources: A6 Section 2 and the stricter shared-account guard. The 5 BC2RL GPU slots (running, pending, array
  concurrency) are shared with stage G. CPU: one job at a time, <= 4 CPUs. Stage-level forecasts in the report.

### 13.11 Dated additions (program S)
- **13.11.1 (2026-10-29, recorded before any st1 data exists; 14.9.2).** The stage-R data `rs1` is drawn before st1.
  The 3-input key universe of the frozen generator has about 18,900 keys (independent count, own audit code); 400,000
  st1 draws take nearly all of it, so rs1 could not be drawn after st1. st1 excludes every earlier ext dataset
  (13.3, unchanged code), so st1 and rs1 stay disjoint. The st1 seed, draw count and rules do not change. The st1
  manifest records the excluded keys.
- **13.11.2 part 1 (2026-10-29 08:15 EDT; S2 development inspection, jobs 10978044 and 10978370; before st1p exists
  and before any S3 run).** Exploratory, dev candidates only (pre_pool, pre_dev; pre_conf only as cell counts).
  - st1: 11,495 candidates (A 1,954, B 9,541). Prescreen (16 attempts, seed 7, light / extended seed-11 adapters):
    pre_pool 958 + 958, pre_dev and pre_conf 471 + 471 each (all class-A candidates of the pool and conf roles).
  - Class is confounded with the target: class A targets have median 14 (p90 34); class B median 120 (p90 345).
    In pre_pool, all 958 A but 245 of 958 B (25.6%) fall in the lowest target bin (1-60).
  - Mean prescreen success, pre_pool: light A 0.097 / B 0.061; extended A 0.315 / B 0.173. Zero-success share:
    light A 24% / B 45%; extended A 3% / B 27%. Mean response tokens: light A 36 / B 50; extended A 20 / B 22.
  - CEM on pre_pool (13.3 cells): 178 matched pairs; 780 A and 780 B unmatched; 19 cells without common support.
    Residual differences of the matched pools (A vs B): target 13.0 vs 36.6; traces 4.0 vs 1.1; input max 15.5 vs
    13.1; success light 0.071 vs 0.062, extended 0.224 vs 0.166.
  - Consequence: 178 pairs allow at most 2N = 128 (64 passes per problem in the A and B arms; 13.4 needs 2N to
    divide 8,192).
  - **Rule added (selection only; 13.3 unchanged):** prescreen every remaining class-B candidate with role pool and a
    target in the lowest bin (ext data `st1x`, panel `pre_pool_x`; same adapters, 16 attempts, seed 7). `s2_build
    build` adds them to pre_pool before CEM. No class-A candidate remains to add. The bins, the CEM cells and the
    matching rule do not change.
  - Part 2 (after the build report, before any S3 run) fixes N, the diagnostic count per cell and the representative
    panel size, and decides gate S2. If the matched pairs stay below 128, gate S2 fails (13.3).
- **13.11.2 part 2 (2026-10-29 09:17 EDT; build job 10978686; before any S3 run).**
  - Panel sizes, fixed at 09:15 before the build report: diagnostic panels at most 128 per class x rate cell;
    representative panels 512 each. Reason: the primary contrasts are within one class, so panel size has priority
    over balance between classes. Cells with fewer candidates are taken whole and reported.
  - st1p: CEM over 3,091 pre_pool candidates (st1s 1,916 + st1x 1,175): **650 matched pairs**; unmatched A 308, B
    1,483; 20 cells without common support. Residual differences (A vs B, all 650): target 14.0 vs 36.2; traces 4.08
    vs 1.04; input max 15.6 vs 13.0; prescreen success light 0.086 vs 0.072, extended 0.254 vs 0.190. The first
    half (325) has the same residuals within 0.5 on every mean. The report states these differences; S3 contrasts
    are within class and do not remove them.
  - sd3d 498 (A 262, B 236); sd3f 496 (A 261, B 235). Cells with fewer than 128: dev A rate bins 0 / 3 (2, 4),
    B bins 0 / 2 / 3 (60, 38, 10); conf A (5, 123, 5), B (70, 33, 4). sr3d 512 (A 110, B 402); sr3f 512 (A 95,
    B 417).
  - **N = 256**: the largest N with 2N <= 650 and 2N dividing 8,192. Arms A and B train on the first 512 of trA / trB;
    the AB arms on the first 256 of each. Every arm: 512 unique problems, 16 passes per problem.
  - **Gate S2: pass.** Both classes have >= 512 matched problems, and the matched cells have common support in
    initial success for both inits (matched cells cover light success bins 0-1 and extended bins 0-2 of 16). Whether 3 inputs give enough shapes for
    S4 is decided at the S4 / S5 choice (13.10), not here.
  - Configs: `scripts/make_s3_configs.py dev --pool 256` (worklist `s3-dev`, 10 runs + 2 initial evaluations, seed
    11). The first S3 run is one canary worker.

## 14. Stage R — retention, policy drift and the source of training responses (owner addition 2026-10-28)

Owner addition A6-R (PROJECT.md, 2026-10-28 ~18:00 EDT). Recorded at 18:04 EDT (commit 49fc6fd), before any stage-R job and before any
stage-R outcome exists. It adds to A6. It does not change stage G (Section 12) or program S (Section 13); those continue.
Question: **does retention depend on how far the policy changes, and on where its successful training responses come
from, beyond differences in initial competence and later improvement?**

**Open item (blocks only the R3 training matrix).** The owner refers to "Section 11 of the updated proposal" for the
common-pool construction and the common supervised loss. The proposal copies in `protocol/` (revision of 2026-10-24)
have no such section. The values that the owner message states are fixed below. The supervised loss and any other
Section-11 detail stay open. No R3 training job is submitted until that text is available and recorded in 14.9. **Closed 2026-10-29 by 14.9.1 (owner definition).**

### 14.1 Stage R1 — policy drift on existing checkpoints (exploratory)
- **Branches:** every A5/A6 online branch with saved adapters (M GRPO and filtered, steps 16/128, seeds 11-66; Y3 MASK
  and T3; Y4 steps 32/64). Reference policy = the branch's own BC checkpoint (step 32 = the re-run adapter, Section
  3.2). Rounds 0 (the reference itself), 64, 128, 256. Only saved files are scored. A missing file is listed as
  missing and is never re-created.
- **Problems:** from the opened dev panels, the 192 yd3 and 64 yd5 problems with the smallest
  sha256("A6-R drift:" + pid). Their existing yd selection, baseline and outcome attempts are reused.
- **Reference responses:** 16 per problem from each BC reference (evaluation decoding, cap 256, sampling seed 8),
  stored with token ids. All responses enter, failures included.
- **Estimator.** KL(initial || current) per problem = mean over its reference responses of
  sum_t [log p_init(y_t | prefix) - log p_cur(y_t | prefix)], summed over every completion token, including the
  sampled stop token. A response that reaches the cap without a stop token contributes its 256-token sum with no stop
  term: the estimate is the KL between the two distributions over capped outputs. Both policies are scored in FP32
  (frozen weights cast exactly; TF32 off). The per-token average (sum / tokens) is a separate diagnostic, not KL.
  Reverse KL is not estimated (it needs current-policy samples).
- **Sampling-scoring check.** The log-prob of each sampled token is recorded during sampling (BF16, as all
  evaluation). Report the signed mean per-token difference (sampling - FP32 scoring of the reference) and the median
  absolute sequence difference. The check passes if |signed mean| <= 0.005 nats/token. If it fails, the KL values
  are reported with that bias bound and the label "sampling/scoring mismatch".
- **Analyses (exploratory; no claim).** Within-seed drift trajectories by round. Per-problem association of round-256
  drift with excess loss_16 (M finals: yd outcome attempts), initial competence (c_a), structure (needs-multiplication,
  number of forms; `bc2rl.structure`), response length and per-problem improvement. Branch-level association with
  the look-1 E values (already opened; descriptive). Retrospective leave-one-seed-out prediction of branch E from
  (competence, improvement) with and without drift. Correlation is not presented as a mechanism.
- After R1, a dated addition names at most two drift relationships to confirm on the R3 runs, with their tests,
  before any R3 confirmation outcome exists.

### 14.2 Stage R2 — retention at similar learning progress
- Progress = pass@1 improvement over the initial adapter on yd3 (dev; 16 attempts per problem per saved checkpoint,
  sampling seed 9). Fixed-round results are always reported; matched-progress results are reported beside them.
- Target level L* (fixed on the dev seed-7 R3 block only, before confirmation): L* = 0.8 x the smallest, over arms
  and inits, of the largest improvement reached at a saved round, rounded down to 0.01. If L* < 0.02, the
  matched-progress comparison is marked unavailable.
- Checkpoint rule (frozen with L*): within each initialization, the earliest saved checkpoint with |improvement - L*|
  <= 0.01 on yd3. A branch without one is unavailable; a seed unit with an unavailable branch is excluded; a
  comparison with fewer than 3 seed units is unavailable. No extrapolation; no use of confirmation outcomes.
- Reported: achieved improvement with its problem-bootstrap interval; for cross-initialization comparisons, matched
  improvement and matched final accuracy are kept separate. Equal observed accuracy is not equal true accuracy.

### 14.3 Stage R3 — response-source experiment (one supervised loss)
| arm | target per prompt occurrence |
|---|---|
| S | the fixed solver trace of the problem (generator reference trace, `task.completion_text`) |
| I | one fixed verified response of the starting BC policy (initial-policy cache, independent attempts) |
| F | a fresh verified response of the same frozen starting policy |
| O | a fresh verified response of this arm's current policy |
Contrasts: S vs I (target source), I vs F (fixed-cache reuse), F vs O (policy refresh). This is not a GRPO-versus-
supervision comparison. The design studies solutions available at the start; it cannot show discovery of new ones.
- **Checkpoints:** dev seed 7 (`bc2rl-s0-bc-devlight` step 16, `bc2rl-s0-bc-devext` step 128); confirmation BC steps 16
  and 128 of seeds 11, 22, 33: 6 four-arm blocks, **24 confirmatory branches**.
- **Common pool (3 inputs; new data `rs1`, seed 20261030; all earlier keys, st1 included, excluded):** candidates in
  draw order are screened with independent attempts under all six confirmation checkpoints; a candidate qualifies
  when every checkpoint solves it at least once. Stop at 256 qualifying problems or 2,048 screened candidates. If
  fewer than 256 qualify, the pool is the first 128 (the development-fixed fallback); fewer than 128 is a failed gate
  (diagnosis, not a relaxed rule). The screening attempt count is fixed in 14.9 before screening. Exclusions and
  screening costs are recorded.
- **Caches:** I = for each starting checkpoint (8), sample with independent attempts until the first verified success,
  at most 64 attempts; a pool problem without an I target for a confirmation checkpoint is excluded and counted.
- **Fresh targets (F, O):** per prompt occurrence, sample until the first verified success or 64 attempts; one target
  per occurrence. If F or O has no target, the occurrence is masked in all four arms. Masking above 5% of the
  occurrences of a block is reported as a failure of the exposure-matching criterion (threshold fixed now).
- **Training:** 256 rounds x 32 prompt occurrences (same prompt order in all arms of a block), learning rate 1e-5,
  adapter, decoding and optimizer settings as the frozen configuration; one optimizer step per round per arm; identical
  accepted occurrences and loss denominator in all arms; a round with every occurrence masked takes no step in any arm.
  Saved rounds 0, 32, 64, 128, 192, 256. One GPU worker runs a block: the four arms in sequence in each round, each
  with its own optimizer state. Every candidate (accepted or not), cache cost and token is counted. **The supervised
  loss: 14.9.1.**

### 14.4 Evaluation and inference (R3; frozen before confirmation)
- Fresh panels from `rs1`: rf3 = 768 three-input, rf5 = 128 five-input problems. Initial adapters: 48 attempts
  (0-23 selection, 24-47 baseline); finals: 32 attempts; estimator, bins and intervals as Section 3.4 (pooled weights
  over the 6 confirmation initial adapters; K = 16).
- **Primary:** the three source contrasts at round 256 (E(X) - E(Y) per init; seed unit = mean over the two inits):
  F vs O (named primary), S vs I, I vs F. Holm correction over the three at family alpha 0.05: a contrast's p-value is
  the larger of the paired problem-bootstrap p and the seed-level t p (df 2); a claim needs the Holm step to pass.
  Problem and seed uncertainty are reported separately.
- Secondary: the initialization interaction of each contrast; matched-progress versions (14.2); whole-panel pass@1
  and the full supported pass@k / pass^k curves; initial, final and change values; drift (14.1 protocol, own
  reference); failure types; target diversity (canonical forms per problem, `bc2rl.structure`); generated tokens;
  actual optimizer steps; masking rate. Fixed-exposure results stay reported when matched-progress results differ.

### 14.5 Order, stopping and resources
- Order: R1 (scoring code + test -> GPU scoring -> CPU analysis) and the R3 development checks (rs1 data, screening,
  caches, dev block seed 7) come first; the 6 confirmation blocks start only after 14.9 records the loss and the
  frozen settings. R2's L* is fixed on the dev block.
- Three seeds are the prespecified confirmation size; no seed is added unless a precision plan is recorded before
  confirmation. An uninformative comparison is closed explicitly.
- Resources: A6 Section 2 and the stricter shared-account guard; the 5 BC2RL GPU slots (running, pending, array
  concurrency) are shared with stage G and program S; one GPU per job; no MPS, no multi-GPU, no general GPUs; CPU one
  job at a time. Planning forecast: R1 about 3-5 GPU-h; R3 screening and caches about 2-4 GPU-h; dev block plus
  evaluations about 4-6 GPU-h; confirmation 6 blocks plus evaluations (incl. R2 dev measurements) about 25-35 GPU-h.

### 14.9 Dated additions (stage R)
- **14.9.1 Supervised loss of R3 (owner instruction A6-R2, PROJECT.md 2026-10-29; recorded before any R3 job).**
  This closes the open item of Section 14. All four arms (S, I, F, O) use
  L(theta) = -(1 / (32 x 256)) x sum_{i=1..32} m_i x sum_{t=1..T_i} log pi_theta(y_it | x_i, y_i<t).
  - y_i: one verified successful response from the arm's source. m_i: the shared acceptance mask (m_i = 0 in all
    four arms when F or O has no verified response within 64 attempts). T_i: response tokens incl. the sampled stop
    token; no prompt or padding tokens. A truncated response is never a successful target.
  - The denominator is fixed at 32 x 256, also when occurrences are masked. It is not the accepted count, the token
    count or a response length.
  - Teacher forcing. Targets and acceptance decisions are fixed data; the gradient passes only through the trained
    policy's log-probabilities. No baseline, advantage normalization, importance ratio, clipping surrogate or KL term.
  - AdamW, lr 1e-5, weight decay 0, gradient-norm clip 1.0 (per arm), default betas and eps, constant learning rate;
    a fresh optimizer state per arm at the branch. Precision and trainable parameters as the frozen configuration
    (BF16 base weights, FP32 LoRA r 16 adapter and optimizer state), identical in all arms. One full-batch step per
    round per arm (micro-batch gradients accumulated). A round with all 32 occurrences masked takes no optimizer step
    in any arm; it counts as a sampling round.
  - Each round samples the F and O targets before any arm is updated, applies their common mask to all four arms and
    then steps the four arms in the order S, I, F, O, each with its own optimizer state. 256 sampling rounds; adapters
    saved at rounds 0, 32, 64, 128, 192, 256.
  - This normalization differs from the A5 success-filtered baseline (1 / (|S| x cap)). R3 is the controlled
    response-source experiment, not a reproduction of that baseline.
- **14.9.2 R3 data, screening, caches and development settings (recorded before any R3 job).**
  - Data `rs1` (`bc2rl xdata`, seed 20261030, every earlier ext dataset excluded): `rsc` 2,048 three-input screening
    candidates (draw order), `rd3` 256 three-input (development panel), `rf3` 768 three-input and `rf5` 128
    five-input (confirmation panels; not read before 14.9.3). rs1 is drawn before st1 (13.11.1).
  - Screening: the 6 confirmation checkpoints, 16 independent attempts per candidate (evaluation decoding, cap 256,
    seed 10), chunks of 256 in draw order; stop after the chunk that reaches 256 qualifying problems, or at 2,048.
    Pool = the first 256 qualifying problems in draw order; else the first 128; fewer than 128 = failed gate.
  - Caches (I): the 6 confirmation and 2 development checkpoints; per pool problem, attempts in batches of 8 until
    the first verified success in attempt order, at most 64 (seed 11). Common pool = pool problems with an I target
    for all 6 confirmation checkpoints. A development block also drops problems without a target for its checkpoint.
    Every attempt and token (also attempts after the first success within a batch) is counted.
  - Fresh targets (F, O): the same rule (batches of 8, at most 64, first verified success in attempt order).
    F and O use different sampling seeds. Prompt stream: 32 occurrences per round from the block pool, stream seed
    = the block seed (7 for development; the BC seed for confirmation).
  - Verified success = verifier reward 1 and a sampled stop token.
  - Development (seed 7, `bc2rl-s0-bc-devlight` step 16 and `bc2rl-s0-bc-devext` step 128): two four-arm blocks;
    evaluations on rd3 (initial adapter 48 attempts, split 24; round-256 arms 32 attempts; seed 6) and R2 progress
    on yd3 (16 attempts, seed 9; initial adapter and every saved round of every arm).
  - **Development gate DR (all must hold before 14.9.3 and any confirmation block):** (1) pool >= 128; (2) both
    development blocks complete 256 rounds with finite losses; (3) masked occurrences <= 5% per block (otherwise:
    diagnosis and a dated decision, no confirmation); (4) every S and I target ends with a stop token and fits the cap;
    (5) measured GPU-h per block and per evaluation give the confirmation forecast. L* and the R2 checkpoint rule
    (14.2) are computed on the development data; L* < 0.02 makes R2 unavailable but does not stop R3.
  - 14.9.3 (after gate DR, before any confirmation block) freezes L*, the confirmation configs and the R1
    relationships to confirm.
- **14.9.3 (2026-10-29 16:08 EDT, written after gate DR and before any confirmation block; dev jobs 10976381 and
  10978682; audit script `audits/scripts/r3_gate_dr.py` in the workspace).**
  - **Gate DR: pass.** (1) pool 256 (334 of 1,024 screened qualified; common pool 255). (2) Both dev blocks: 256
    rounds, 256 optimizer steps per arm, no skipped round, all 1,024 logged losses finite. (3) Masked occurrences:
    b16 356 / 8,192 = 0.0435; b128 159 / 8,192 = 0.0194 (limit 0.05). (4) `rsource` stops before training if an S
    or I target exceeds the cap or lacks a stop token; both blocks trained. (5) Measured cost below.
  - **L* = 0.07.** The smallest, over arms and inits, of the largest yd3 pass@1 improvement at a saved round is
    0.0973 (b128 S, round 128); 0.8 x 0.0973 = 0.078, rounded down to 0.07. R2 is available.
  - **Checkpoint rule (frozen as 14.2):** within each init, the earliest saved checkpoint with |improvement - L*| <=
    0.01 on yd3. On the dev block it finds a checkpoint for 2 of 8 branches (b128 F round 32; b128 S round 256). The
    b16 arms improve by 0.19-0.22 by round 32, the first saved round. Most matched-progress comparisons are therefore
    expected to be unavailable. The save grid (0 / 32 / 64 / 128 / 192 / 256) is the owner's (14.9.1) and does not
    change. The fixed-round results (14.4 primary) do not depend on this rule.
  - **Confirmation configs** (`scripts/make_r_configs.py rconf`; worklist `r3-conf`, 24 tasks; test
    `test_r3_conf_generated_configs`). Six blocks: BC steps 16 and 128 x seeds 11, 22, 33. Each block equals the dev
    block of its init except: starting checkpoint = the seed's BC adapter (sha256 checked against the rprep cache),
    I targets from that checkpoint's cache, stream and sampling seed = the BC seed. Pool = the common pool (255).
    Per block, three further tasks after the block's summary exists: rf3 + rf5 evaluation (initial adapter 48
    attempts, split 24; round-256 arms 32 attempts; seed 6); yd3 R2 progress (16 attempts, seed 9; initial adapter
    and rounds 32-256 of every arm); drift (14.1 protocol: the 256 R1 problems, the block's own BC reference
    responses from R1, rounds 64 / 128 / 256 of every arm). The tasks are separate so that each fits one 8-h job.
  - **R1 relationships to confirm (at most two; R1 was exploratory, Section 14.1).** Drift = KL(initial ||
    current) per sequence at round 256 on the R1 problems, as in 14.1. Seed unit = the mean over the four arms.
    - D1 (init): drift(b16 branch) - drift(b128 branch) > 0, per seed and arm. R1: b16 > b128 in all four GRPO
      branch types (M, filtered, MASK, T3) at round 256.
    - D2 (competence, b128): over problems, the partial Spearman correlation of drift with the initial selection
      count c_a (yd rows of the b128 BC checkpoint), given the mean reference-response length, is < 0. R1: -0.24
      to -0.30 at b128, 0 of 6 seeds positive (GRPO and filtered M).
    - Test: seed-level t (df 2, two-sided) over the 3 seed units; Holm over D1 and D2 at family alpha 0.05. A
      relationship is confirmed only if its Holm step passes and all 3 seed units have the predicted sign. The
      per-token diagnostic is reported beside each, as a secondary value. These are statements about
      association in R3 supervised training, not about mechanism.
  - **Measured cost and forecast (planning only).** Dev: rprep 1.24 GPU-h; block training 4.48 (b16) and 2.86
    (b128) GPU-h; yd3 progress evaluation 0.94 / 2.94 h; rd3 evaluation 0.42 / 1.00 h. The rf panels hold 896
    problems against 256 in rd3. Forecast per confirmation seed: about 7-8 GPU-h (b16) + 9-10 GPU-h (b128); for
    the 6 blocks about 45-55 GPU-h, against 25-35 in 14.5. The order of work does not change; the 5-slot cap
    applies. One block worker runs first as a canary.
