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
