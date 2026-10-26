# BC2RL extension study (amendment A5)

**Status: update 1, 2026-10-26 08:20 EDT (12:20 UTC).** No extension result exists yet. One of five X5 evaluation
workers is running, and four wait in the preempt queue. Update 0 (2026-10-25 23:49 EDT) gave the design and the
forecast. Updates follow at least every 8 hours.

## 0. Changes since update 0
| item | observation (source) |
|---|---|
| Fresh panels built | Data job 10933044 COMPLETED at 00:00 EDT, with 57 of 57 tests passed (`results/bc2rl-x5-data-main-s0`). xc3 = 768 3-input problems (sha256 `861dcebd...`, all chain). xc5 = 192 5-input problems (sha256 `fe132ec6...`, 135 branch, 57 chain). |
| Panel coverage limit | For 3 inputs, 200 of 968 accepted draws (20.7%) had a key already used in a v1 split and were rejected. xc5 had 0 rejections. Frequent 3-input keys are more often rejected. xc3 therefore under-represents the most frequent part of the generator distribution. X5 results apply to the fresh panel, not to the full generator distribution. |
| First failures | Data job 10932942 failed on a test assumption. Test job 10932995 failed on an import-order error in `bc2rl/mech.py`. Commit `e1f15cd` fixed both, and the tests passed in job 10933044. No result-producing code changed. |
| Preemption | Worker w1 (job 10932944) started 07:22 EDT and was preempted at 07:34 EDT. Its log has no `[xeval]` result line, so no (adapter, panel) result was complete. Slurm requeued it. |
| Running | Worker w2 (job 10932945) started 08:13 EDT on one L40S. |
| Recipe defect (no effect on outcomes) | The X5 worker recipes use `"true"` as the done-predicate of the `precondition` phase. Therefore the GPU-memory precondition never runs. It is a safety check only and cannot change a count. The submitted jobs keep their inlined scripts. The X6 recipe now uses `"false"`, so the check runs. |

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
| X5 confirmation | 6 BC + 12 round-256 adapters on 768 fresh 3-input and 192 fresh 5-input problems | panels built; evaluation 1 of 5 workers running |
| X6 E2 repair | fp32 measurement, radius grid 1/16-1 step, validity checks V1-V5 on 2 dev checkpoints; study run only if they pass | code tested (job 10933044); GPU job waits for a free slot |
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
| 10932942 (FAILED), 10933044 (COMPLETED) | bc2rl-x5-data-main-s0 (tests + panels) | CPU, 4 cores | — |
| 10932944, 10932945, 10932951, 10932954, 10932955 | bc2rl-x5-evalw1..w5 | 1 preempt GPU each (5 = cap) | data |
| 10932995 (FAILED; superseded by the tests in 10933044) | bc2rl-x6-tests-main-s0 | CPU | — |
| not yet submitted | bc2rl-x6-calib-dev-s7 | 1 preempt GPU | a free GPU slot (after an X5 worker ends) |
| not yet submitted | bc2rl-x5-an-main-s0 (analysis, once) | CPU | all 5 X5 workers |

## 6. Timeline and resource forecast

**Use to date (sacct, 2026-10-26 08:17 EDT).** GPU use before A5: 13.80 allocated GPU-h. A5 so far:
| job | resources | elapsed | GPU-h |
|---|---|---:|---:|
| 10932944 (w1, preempted attempt) | 1 L40S | 716 s | 0.20 |
| 10932945 (w2, running) | 1 L40S | 219 s at the time of the check | 0.06 |
| 10932942, 10932995, 10933044 | 4 CPU each | 603 + 14 + 289 s | 0 (1.0 CPU-h) |
| **A5 total** | | | **0.26 GPU-h** (cumulative with A0-A4: 14.06) |

A5 has no GPU-hour cap. The work forecast from update 0 does not change:

| stage | work | GPU-h forecast | wall after start (5 GPUs) |
|---|---|---:|---|
| X5 evaluation | 645,120 completions at a measured 22-40 completions/s | 4.5-8 | 1-2 h per worker |
| X5 analysis | CPU, 10,000 bootstrap resamples | 0 | 0.5 h |
| X6 calibration | 2 dev checkpoints, 12 directions x 5 radii, fp32 | 0.5-1 | 1 h |
| X6 study run (if valid) | 6 checkpoints x 8 pairs; probe 48 x 15 per fork | 3-5 | 1-1.5 h |
| X7 (if launched) | per new seed: BC path + 4 branches of 256 rounds (measured 1.9 GPU-h each on A6000) + evaluation | about 8 per seed | 3-5 h per batch of 5 |

**Queue (squeue, 08:17 EDT).** The preempt partition has 363 pending and 276 running jobs. The queue wait was
longer than the update-0 median: w2 waited 8.2 h after its data dependency cleared (00:00:39 to 08:13:42 EDT). Slurm start estimates for the
pending workers are 08:26 to 12:14 EDT. These estimates are not reliable, and preemption can delay them.

**Revised completion estimates (replace update 0).**
| milestone | estimate (EDT, 2026-10-26) |
|---|---|
| all 5 X5 workers complete | 12:00-20:00 |
| X5 analysis and decision | 1 h after that |
| X6 calibration submitted | when w2 ends (about 09:30-10:30); starts when the queue allows |
| X6 gate decision | 12:00-22:00 |
| X6 study run (if the gate passes) and X7 decision | 2026-10-27 |

## 7. Blockers

None. The preempt queue is the main source of delay. The work stays within the 5-slot cap.

## 8. Observations, interpretations, hypotheses (this update)
- Observation: the X5 panels exist and meet their pre-registered sizes (768 and 192). No X5 or X6 outcome exists yet.
- Interpretation: the 20.7% duplicate rejection on 3 inputs limits the population that X5 can describe (Section 0).
  It does not bias the extended-versus-light comparison, because all 18 adapters see the same problems.
- Hypothesis under test: unchanged (Section 1). No new hypothesis.

## 9. Next experiments
1. Finish X5 evaluation (5 workers). Check each worker: 18 adapters x 2 panels, adapter sha256, 48/32 attempts.
2. Submit `bc2rl-x6-calib-dev-s7` when a GPU slot frees.
3. Run the X5 analysis once, after all 5 workers complete.
4. X7: decide after X5 and X6. Pre-register before any launch.
