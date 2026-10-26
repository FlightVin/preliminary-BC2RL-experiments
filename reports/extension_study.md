# BC2RL extension study (amendment A5)

**Status: update 0, 2026-10-25 23:49 EDT (2026-10-26 03:49 UTC).** No extension result exists yet. This update gives
the design, the submitted jobs, the timeline and the resource forecast. Updates follow at least every 8 hours.

Protocol: `protocol/AMENDMENT-A5-2026-10-26.md` (pre-registration, committed before any extension job: X5 at `b2ad826`,
X6 at `eb19ea8`). The initial study stays frozen (`reports/initial_study.md`); its audit corrections are in its Section 12.

## 1. Question

Does extended BC (128 steps) make intermittently successful problems more vulnerable to online training than light BC
(16 steps), at the same initial success probability?

The question comes from a **post hoc** observation in the initial study (Section 8.2: on test, at the same initial count
of 15, extended-BC branches lost more 3-input problems). That observation is not evidence here. This extension tests it
on new problems.

## 2. Stages

| stage | what | status |
|---|---|---|
| X1 audit | recount the frozen report's band checks, pass@1 / pass^1 duplication, retention and band scope | **done** (project `6d70485`; `audits/report-audit-20261026.md`) |
| X5 confirmation | 6 BC + 12 round-256 adapters on 768 fresh 3-input and 192 fresh 5-input problems | submitted |
| X6 E2 repair | fp32 measurement, radius grid 1/16-1 step, validity checks V1-V5 on 2 dev checkpoints; study run only if they pass | code done and tests submitted; GPU job waits for a free slot |
| X7 training replication | new BC seeds + 4 branches each | conditional on X5/X6; pre-registered before any launch |

### X1 audit result (frozen report; no number changed)
- Test GRPO vs filtered: **4 of 18** distinct checks exclude zero, not "5 of 24". The old count included pass^1, which
  is the same estimate as pass@1.
- Validation GRPO vs filtered: 1 of 18 checks excludes zero. The claim that two exceptions point in opposite
  directions was wrong: the second band has a lower limit of 0.000.
- Retention: the "no loss" statement covers 13 problems of one initial adapter (extended-BC seed 33). Its two branches
  share those problems. The data do not show retention in general.
- Band scope: bands are simultaneous over k within one branch and one metric family. They resample problems with
  the trained adapters fixed, so they do not cover training-seed variability.

## 3. X5 design (pre-registered, protocol Section 3)

- **Fresh problems:** generator v1, new seed 20261026. Any canonical key that occurs in a v1 split, the format example or
  the mechanism splits is rejected. Panels: xc3 = 768 problems with 3 inputs; xc5 = 192 problems with 5 inputs.
- **More initial attempts, separate roles:** each initial adapter gets 48 attempts per problem. Attempts 0-23
  *select* the subgroup; attempts 24-47 estimate the *baseline*. Final adapters get 32 fresh attempts.
- **Why the split matters:** a synthetic check (`tests/test_xanalysis.py`) shows that if the same attempts select the
  problems and also give their baseline, the baseline loss is biased by about -0.11 (regression to the mean). With
  separate sets the bias is zero by construction.
- **Primary statistic:** the population is 3-input problems with 1-11 selection successes of 24, in bins 1-2 / 3-5 / 6-11.
  For each problem, the excess loss is loss_16(final) - loss_16(baseline), where loss_16 is the unbiased estimate of
  (1-p)^16. The bin means are standardized to common weights. The statistic is the extended minus light difference,
  averaged over the 6 (method, seed) pairs.
- **Decision rule:** CONFIRMED if the 95% paired problem-bootstrap lower limit is > 0 and at least 5 of 6 pairs are
  positive. CONTRADICTED if the upper limit is < 0. Otherwise NOT CONFIRMED. One look; no added problems or attempts.
- **Secondary analyses (fixed in advance):** per method; K = 4 / 8 / 24; selection and baseline roles swapped; fine
  bins; bins whose baseline rates match within 0.02; consistency (p^K) change; per-bin baseline rates; the consistent
  subgroup; whole-panel pass@k / pass^k with bands; 5-input coverage next to 3-input loss; per-seed results.
- **Scope:** X5 tests generalization over *problems* for the same trained adapters. It is not a training replication.

## 4. X6 design (protocol Section 4)

The A2 assay failed for two reasons: the BF16 measurement floor, and a large negative curvature term at one AdamW step.
X6 measures the anchor direction and every log-likelihood in fp32. It tests radii of 1/16, 1/8, 1/4, 1/2 and 1 step on
two development checkpoints. A radius is valid only if all five checks pass on both checkpoints:
- V1: repeat noise.
- V2: absolute first-order error of each pair contrast <= 25%, and median relative error <= 25%.
- V3: quadratic term <= 20% of the linear term.
- V4: random-direction control ratio in [0.8, 1.25].
- V5: median sampled-token KL <= 0.01.

If no radius passes, X6 ends with a documented failure. The study run (8 matched pairs on each of 6 checkpoints) is
pre-registered in Section 9 of the protocol before it starts. Whether alignment predicts held-out changes is reported on
development data as a diagnostic; it is not a gate.

## 5. Jobs (Slurm is the authority for state)

| job | expid | resources | depends on |
|---|---|---|---|
| 10932942 | bc2rl-x5-data-main-s0 (tests + panels) | CPU, 4 cores | — |
| 10932944, 10932945, 10932951, 10932954, 10932955 | bc2rl-x5-evalw1..w5 | 1 preempt GPU each (5 = cap) | data |
| 10932995 | bc2rl-x6-tests-main-s0 | CPU | data (one CPU job at a time) |
| not yet submitted | bc2rl-x6-calib-dev-s7 | 1 preempt GPU | a free GPU slot (after an X5 worker ends) |
| not yet submitted | bc2rl-x5-an-main-s0 (analysis, once) | CPU | all 5 X5 workers |

## 6. Timeline and resource forecast

GPU use before A5: 13.80 allocated GPU-h (sacct). A5 has no GPU-hour cap; accounting continues.

| stage | work | GPU-h forecast | wall after start (5 GPUs) |
|---|---|---:|---|
| X5 evaluation | 645,120 completions at a measured 22-40 completions/s | 4.5-8 | 1-2 h |
| X5 analysis | CPU, 10,000 bootstrap resamples | 0 | 0.5 h |
| X6 calibration | 2 dev checkpoints, 12 directions x 5 radii, fp32 | 0.5-1 | 1 h |
| X6 study run (if valid) | 6 checkpoints x 8 pairs; probe 48 x 15 per fork | 3-5 | 1-1.5 h |
| X7 (if launched) | per new seed: BC path + 4 branches of 256 rounds (measured 1.9 GPU-h each on A6000) + evaluation | about 8 per seed | 3-5 h per batch of 5 |

Queue: for the account over the last 72 h, the median preempt wait was 0.33 h and the p90 was 5.5 h; the mean attempt
lasted 2.96 h before a preemption (`bin/ae-capacity`, 2026-10-26). Expected order: X5 results 3-8 h from now; X6 gate
4-10 h; X6 study run (if valid) and the X7 decision after that.

## 7. Blockers

None.
