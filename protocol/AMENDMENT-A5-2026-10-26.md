# Amendment A5 — pre-registered extension (2026-10-26)

Purpose: define the extension study authorized by owner amendment A5 (PROJECT.md, 2026-10-26). This file is committed
before any extension outcome exists. The frozen initial study (`reports/initial_study.md`, result set `22e6816`) stays
frozen; its exploratory and post hoc labels stay. The extension report is `reports/extension_study.md`.

Each stage below is pre-registered in its own section before its outcomes exist. A later section may add a stage; it
may not change an earlier stage after that stage's outcomes exist. Dated clarifications go to Section 9.

## 1. Question and claim boundaries

Question: does extended BC (128 steps) make intermittently successful problems more vulnerable to online training than
light BC (16 steps), at the same initial success probability?

Origin: the post hoc observation of `reports/initial_study.md` Section 8.2 (test split, read once). That observation is
not evidence for this extension; this extension tests it on new problems.

Two kinds of replication are distinct and are reported separately:
- **Fresh-problem confirmation (Stage X5):** the same 6 BC checkpoints and 12 round-256 endpoints, new problems. It
  tests whether the observation generalizes over problems for these trained adapters. It cannot show that the effect
  generalizes over training runs.
- **Training replication (Stage X7, conditional):** new training seeds. Only it addresses training-run variability.

## 2. Resources (owner A5; binding)
- GPUs only on `preempt` / `preempt_qos`; at most 5 BC2RL GPU slots, counting running and pending allocations and
  array concurrency; `tools/bc2rl_guard.py submit` before every submission. No general GPUs, `ae-topup`, standby GPUs,
  or interference with other projects. CPU-only jobs: one at a time on `preempt_cpu_qos`, <= 4 CPUs.
- No cumulative GPU-hour cap. Every allocated GPU-hour is accounted (sacct). The forecast is updated before each stage.
- Scientific comparisons use fixed steps, rounds and attempt counts, never equal GPU-hours.

## 3. Stage X5 — fresh-problem confirmation (pre-registered)

### 3.1 Data
- Generator: frozen v1 configuration (`configs/data_v1.yaml`: inputs 2-20, ops + - *, target <= 1000, |intermediate|
  <= 10,000), new seed **20261026**. Panels: **xc3 = 768 problems with 3 inputs; xc5 = 192 problems with 5 inputs**.
- Disjointness: canonical key `(sorted inputs, target)` excluded if it occurs in any v1 split (dev, devdemo, bc,
  online, val, test), the format example, or the mechanism splits. The count of duplicate rejections is reported: it
  measures how far the fresh panel differs from the generator distribution (frequent keys are more often excluded).
- Command: `python -m bc2rl xdata --config configs/ext/xdata.json` -> `$PROJ_DATA/data/v1/ext/x5/`.

### 3.2 Adapters and sampling
- Initial: light (step 16) and extended (step 128) BC adapters of seeds 11, 22, 33 (6 adapters). **48 attempts per
  problem**: attempts 0-23 = **selection** set (c_a), attempts 24-47 = **baseline** set (c_b).
- Final: the 12 round-256 E3 adapters. **32 fresh attempts per problem** (c_f).
- Decoding, prompt, cap and verifier exactly as frozen (temperature 1, top-p 1, cap 256, format example, verifier v1).
  Adapter sha256 must equal REGISTRY values; the job stops otherwise.
- Every adapter is evaluated on both panels. Sampling seed per (adapter, panel): `seed * 1000003 + stable_int(tag:panel)`.

### 3.3 Estimators
For n independent attempts with c successes: loss_k(n, c) = C(n-c, k)/C(n, k) (unbiased for (1-p)^k; = 1 - pass@k)
and rel_k(n, c) = C(c, k)/C(n, k) (unbiased for p^k; = pass^k). Selection uses c_a only. Baseline and final estimates
use attempts that played no part in selection, so the selected-set baseline has no regression-to-the-mean bias.
Initial-to-final comparisons use k <= 24 (baseline n = 24). Whole-panel curves use k <= 48 (initial) and k <= 32 (final).

### 3.4 Primary analysis (one test)
- Population: xc3 problems with **1 <= c_a <= 11 of 24** ("intermittent"), in bins B1 = 1-2, B2 = 3-5, B3 = 6-11.
- Per problem: excess loss e = loss_16(32, c_f) - loss_16(24, c_b) (**K = 16**).
- Per branch (init i, method m, seed s): E = sum_b w_b mean_{x in b}(e) / sum_b w_b, where w_b is the pooled share of
  bin b over all 6 initial adapters (selection counts only).
- Bin exclusion rule (selection counts only): a bin enters the (m, s) comparison only if both the light and the
  extended adapter of seed s have >= 10 problems in it. Weights renormalize over the bins that enter.
- Per pair: Delta(m, s) = E(extended, m, s) - E(light, m, s). **Primary statistic: mean of the 6 Deltas.**
- Uncertainty: paired whole-problem bootstrap, 10,000 resamples of xc3 problems (same resampled problems for all 18
  adapters; bins fixed on the observed selection counts); percentile 95% interval.
- **Decision rule:** CONFIRMED if the 95% lower limit > 0 **and** Delta(m, s) > 0 in >= 5 of 6 pairs. CONTRADICTED if
  the 95% upper limit < 0. Otherwise NOT CONFIRMED. If the upper limit is < 0.05, the report also states "any effect is
  smaller than 0.05 on this scale".
- Interpretation scope: the interval covers problem sampling for these 6 + 12 fixed adapters. It does not cover
  training-seed variability. Equal selection bins do not establish equal true success probability; Section 3.5 reports
  the baseline rate in each bin for both inits.

### 3.5 Pre-registered secondary and sensitivity analyses (reported with intervals; no decision)
1. Per method: mean Delta over seeds for GRPO and for filtered supervision.
2. K = 4, 8, 24 (same bins).
3. Role swap: select on c_b, baseline from c_a.
4. Fine bins: each single c_a value 1..11 (same >= 10 rule).
5. High bin added: B4 = 12-23 (where available).
6. Baseline-matched bins: only primary bins whose mean baseline rate differs by <= 0.02 between light and extended in
   every seed (decided from initial attempts only).
7. Consistency: the same statistic with rel_16 (change in p^16) on the intermittent bins; Delta(K) curves for loss and
   for consistency, K = 1..24, with a 95% band simultaneous over K.
8. Per-bin table: n, selection rate, baseline rate, final rate, final and baseline loss, excess, for every adapter.
9. Initially consistent problems (c_a >= 22/24): baseline vs final pass^k, k in {1, 4, 8, 16, 24}, where n >= 12.
10. Whole panels (xc3, xc5): pass@k and pass^k for every adapter over the supported k; paired final - initial
    contrasts k = 1..32 with pointwise and simultaneous (over k, per branch and family) bands.
11. 5-input coverage: change in pass@k (k = 1, 4, 16, 32) per branch on xc5, listed next to the branch's 3-input
    excess loss. Descriptive association only (12 branches share 6 initial adapters).
All results are shown per seed. Bands are problem-bootstrap bands, conditional on the trained adapters.

### 3.6 Exclusions, stopping and integrity
- No problem is excluded. Truncated, malformed and illegal outputs are failures.
- Fixed sample: one look, after all 18 adapters x 2 panels are complete. No extra problems or attempts after any
  outcome is seen. An incomplete or failed evaluation is rerun with the identical configuration and seed.
- The analysis code (`bc2rl/xanalysis.py`, `scripts/x5_confirm.py`) is committed and tested on synthetic data before
  any evaluation job is submitted. It refuses to run on an incomplete set.

### 3.7 Sample size (from measured throughput and the planning effect)
- Planning effect (from the post hoc test data, not a result of this stage): loss rates at c = 1-4 of 15 differed by
  0.05-0.21 between inits. With xc3 = 768, about 200-300 intermittent problems per adapter are expected; the per-pair
  standard error of Delta is about 0.04 and of the mean of 6 pairs about 0.02-0.03 (shared problems correlate the
  pairs). A true Delta of 0.08 then gives a high probability of a lower limit > 0 and >= 5 positive pairs.
- Cost: 6 x 960 x 48 + 12 x 960 x 32 = 645,120 completions. Measured evaluation throughput on Babel preempt GPUs:
  22-40 completions/s (mixed input counts; L40S 21.9, A6000 23.5, RTX 6000 Ada 25.5-29.2, Blackwell 34.0, A100 39.9;
  harvested `eval_*.json`, 2026-10-24/25); 3-input completions are shorter (mean 21-133 tokens). Forecast 4.5-8 GPU-h
  in 5 jobs, about 1-2 h wall when the GPUs start.

## 4. Stage X6 — E2 repair (pre-registered gates; development calibration first)

### 4.1 What failed and what changes
The A2 E2 calibration (dev light-BC seed 7) failed its Taylor check at rho = 1 (one AdamW step): a BF16 measurement
floor and a large negative curvature term of GRPO displacements on the anchor log-likelihood (FINDINGS 2026-10-25
11:15/11:30). Rule 2 of A2 Section 11 stopped E2. That outcome stays in the initial report. X6 is a new, separately
labelled assay. Changes: every log-likelihood, the anchor direction h and the Taylor checks use an **fp32** forward
pass; the radius grid extends **below one step**: rho in {1/16, 1/8, 1/4, 1/2, 1} (displacement = rho x the actual
one-step AdamW displacement, norm-equalized within the pair). Selection, matching, KL statistic and probe sampling are
as frozen in A2 Section 4.

### 4.2 Development calibration (dev light-BC s7 and dev extended-BC s7; dev data and anchor-fit only)
Per dev checkpoint: 4 matched high/low pairs (8 displacements) and 4 random sign directions of equal norm. For each rho:
actual change of the anchor-fit and of a dev held-out reference-trace log-likelihood (dev split, 48 problems per input
count, reference traces), each measured twice (fp32 repeat).
Validity checks for a radius rho (all must hold on **both** dev checkpoints):
- V1 noise: fp32 repeat |difference| <= 0.05 x median |predicted change| (h.d x rho).
- V2 first-order accuracy, absolute: for every pair, |(actual_H - actual_L) - (pred_H - pred_L)| <= 0.25 x
  |pred_H - pred_L| on anchor-fit; and the median over displacements of |actual - pred| / |pred| <= 0.25.
- V3 curvature: the fit actual(rho) = a rho + b rho^2 over the grid; at the chosen rho, |b| rho^2 <= 0.2 |a| rho for
  >= 7 of 8 displacements.
- V4 random-direction control: actual / predicted in [0.8, 1.25] for all 4 random directions at the chosen rho.
- V5 KL: median sampled-token KL <= 0.01 (A2 rule; holds for rho <= 1 from the A2 calibration, re-measured).
Chosen radius: the **largest** rho in the grid that passes V1-V5 on both dev checkpoints. If none passes, X6 ends:
"E2 assay not valid at any tested radius", with the data. The grid, thresholds and sets are not changed afterwards.
Diagnostic only (no gate): on the dev held-out set, the sign agreement between predicted and actual H - L contrasts,
and the ratio actual / predicted. Gating on it would select on the outcome of interest.

### 4.3 Study run (only if 4.2 passes; its pre-registration is completed in Section 9 before it starts)
6 study checkpoints (light/extended x seeds 11/22/33), **8 matched pairs each** (A2 allowed 8 "if resources permit";
no GPU-hour cap now). Per checkpoint, V1-V3 are re-checked on anchor-fit only; a checkpoint that fails is reported
as invalid, not re-tuned. Endpoints as A2 Section 4: primary mechanistic = paired H - L change in anchor-check
log-likelihood (fp32); behavioural = paired change in probe pass@1 (15 attempts; full pass@k / pass^k). Selection uses
anchor-fit and source-pool statistics only; anchor-check and probe never enter selection or radius choice. Light vs
extended comparison of the H - L contrast is reported per seed with pair-level and checkpoint-level summaries.

## 5. Stage X7 — training replication (conditional; pre-registered before launch)
Considered only after X5. New BC seeds and the 4 branches per seed, with the frozen protocol, would test whether the
X5 result holds over training runs. Its launch, seeds, endpoints and decision rule are written in Section 9 before any
new training starts, with a measured forecast. Stop rule: do not launch when X5 and X6 leave no question that new
seeds would change materially.

## 6. Monitoring and reporting
Scheduler events through `ae-watch` (background); batched checks; no polling loops. `reports/extension_study.md` is
updated at least every 8 hours. Before each update: code audit (implementation; statistics/protocol) and writing
review (evidence; clarity), recorded in the harness `audits/` directory.

## 9. Dated clarifications

### 9.1 (2026-10-26, before any X6 job or outcome) X6 calibration inputs and code
- Source pool for the development calibration: `devdemo` + `online` prompts (1,024 drawn with the A2 pool seed). The
  `dev` split is the held-out reference-trace set of Section 4.2 and is therefore not in the source pool. Reason: the
  A2 calibration pool (dev + devdemo, 656 prompts) gave only 73 mixed groups and 2 feasible pairs for dev light-BC.
- "4 matched pairs" means up to 4: the pairs that the frozen pool/matching rules make feasible (at least 2 required;
  otherwise the calibration of that checkpoint fails). V3 with n displacements requires n - 1 to pass.
- Random directions: 4 sign vectors scaled to the median norm of the equal-norm displacements (seed 12345).
- Implementation: `bc2rl/mech.py` options `fp32_measure`, mode `calibrate2`, `validity_check`; gate `scripts/x6_gate.py`;
  configs `configs/ext/x6-calib-{devlight,devext}.json`; tests in `tests/test_mech.py`.

### 9.2 (2026-10-26 ~10:15 EDT, after X6 attempt 1, before any V1-V5 value exists) Calibration attempt 1 failed on pair feasibility
- Attempt 1 (job 10937583, commit `9325386`, run `bc2rl-x6-calib-devlight-s7`) stopped with an exception before any
  validity check. Its selection record (job 10937679, `seldiag.json`) shows the cause. The dev light-BC adapter solved
  8.1% of 3-input attempts, 0.3% of 5-input attempts and 0% of 10-input attempts. Therefore only 98 of 1,024 source
  prompts had a mixed group. The pool rule made 6 pools of 16-17 groups. 5 pools had an infeasible composition, and the
  6th failed the length tolerance (0.066 > 0.05). Feasible pairs: 0. The dev extended-BC calibration did not run.
- Under Section 9.1 this counts as a **failed calibration of attempt 1** (fewer than 2 feasible pairs). It is reported.
- Cause: an input-sizing error. More source prompts produced more pools, not larger pools. No V1-V5 threshold, rho
  grid, gate rule, pair-matching rule or held-out set is involved.
- Attempt 2 (pre-specified here): the source pool becomes 2,048 prompts (same splits, same pool seed; 4,608 available).
  Expected: about 196 mixed groups, 8 pools of about 24. The new runs are `bc2rl-x6-calib-{devlight,devext}-s7-v2`; the
  attempt-1 files stay on disk. All other settings are unchanged. Attempt 2 is the last pool-size change: if a checkpoint
  again has fewer than 2 feasible pairs, X6 stops with a documented failure ("assay not constructible").
- Code: a checkpoint with fewer than `min_pairs` = 2 feasible pairs now writes a failure record (`status`
  `insufficient_pairs`, every rho fails) instead of an exception. The gate then fails. Tests: `tests/test_mech.py`
  (`test_insufficient_pairs_record_fails_gate`, `test_x6_calibration_configs`).

### 9.3 (2026-10-26 ~11:30 EDT, after the X5 result, before any X7 job) Stage X7 pre-registration: training replication
**Why X7 runs (Section 5 stop rule).** X5 confirmed the effect on fresh problems, but for the same 3 training seeds per
initialization. The per-pair Delta ranged from 0.090 to 0.338, and E_light from 0.030 to 0.276. X5 cannot show whether the
effect holds over new training runs. New seeds can change that conclusion, so the stop rule does not apply.

**Design.** New seeds 44, 55, 66. For each seed: the frozen broad-BC path (`configs/main/bc-broad-s33.yaml` with only
`seed`/`run_id` changed; checkpoints at 16 = light, 128 = extended), then the 4 frozen online branches
({light, extended} x {GRPO, filtered}; the seed-33 configs with only `seed`, `run_id`, `stream_seed` and the init path
changed; 256 rounds). Endpoint: round 256 (as X5). Configs: `configs/ext/x7/`, worklist `configs/worklists/x7.json`;
`tests/test_xanalysis.py::test_x7_configs_match_frozen_protocol` checks them against the frozen configs.

**Evaluation.** The X5 fresh panels xc3 (primary) and xc5, with the X5 protocol unchanged (`bc2rl xeval`, initial
adapters 48 attempts with selection 0-23 / baseline 24-47, finals 32 attempts, seed 5, cap 256). The panels are
fresh for the new adapters: no training or selection step used them.

**Primary statistic and decision (one look).** The X5 primary statistic (Section 3.4) over the 6 new (method, seed)
pairs, with weights pooled over the 6 new initial adapters; 10,000 paired problem-bootstrap resamples. Seed-level
Delta_s = mean of Delta(GRPO, s) and Delta(filtered, s); a seed without a usable pair counts as not positive.
- **REPLICATED** if the 95% lower limit > 0 **and** Delta_s > 0 for all 3 new seeds.
- **CONTRADICTED** if the 95% upper limit < 0.
- **NOT REPLICATED** otherwise.
The seed condition is the training-replication test; the bootstrap interval covers problem sampling only.

**Secondary (descriptive).** Per-pair and per-method Delta; swap roles; K = 4, 8, 24; consistency p^16 change; per-bin
table; whole-panel final - initial contrasts on xc3 and xc5. Six-seed summary: Delta_s for seeds 11, 22, 33 (X5
adapters) and 44, 55, 66, their mean and a Student-t 95% interval (df = 5), and the count of positive seeds.

**Exclusions and integrity.** No problem is excluded. Truncation counts as failure. A branch that fails deterministically
is reported and its pairs count as not positive; it is not re-run with changed settings. A preemption resumes from
the last checkpoint (frozen online code). The analysis (`scripts/x7_replicate.py`) refuses an incomplete adapter set.

**Stop rule.** X7 is the last training stage of A5. After X7 the extension ends with a final report, whatever the
outcome. No further seeds are added to change a decision.

**Forecast (measured).** BC: 5.5 min per seed on A100 (jobs 10916268/70). Online: about 1.9 GPU-h per branch (frozen
study). Evaluation at 21.5-66.8 completions/s (X5): about 0.35 GPU-h per final, 1.1 GPU-h per seed for the two initial
adapters. Total about 31 GPU-h: BC 0.3, online 22.8, evaluation 7.5. Jobs: 1 BC job (3 seeds), then 5 workers
(at most 5 GPU slots including pending). Wall time after the workers start: about 6-7 h, plus queue waits.
