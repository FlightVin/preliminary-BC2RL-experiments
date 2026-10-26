# BC2RL extension study (amendment A5)

**Status: update 2, 2026-10-26 11:10 EDT (15:10 UTC).** X5 is complete: its pre-registered primary result is
**CONFIRMED** on fresh problems. X6 is complete: the repaired E2 assay **failed** its pre-registered gate, so the E2
study run does not take place. X7 (training replication on new seeds) is pre-registered and submitted; it has no result
yet. Updates follow at least every 8 hours. Update 0: 2026-10-25 23:49 EDT; update 1: 2026-10-26 08:20 EDT.

Protocol: `protocol/AMENDMENT-A5-2026-10-26.md` (X5 pre-registration `b2ad826`; X6 `eb19ea8`, Sections 9.1-9.2;
X7 Section 9.3, `b65d9cd`). The initial study stays frozen (`reports/initial_study.md`; corrections in its Section 12).
Audit of this update: `audits/extension-audit-20261026T1510Z.md` in the harness workspace.

## 0. Summary

| stage | result | evidence |
|---|---|---|
| X1 audit | 4 count/scope errors corrected in the frozen report; no number changed | Section 2 |
| X5 confirmation (fresh problems, same adapters) | **CONFIRMED**: Delta = +0.203, 95% CI [0.171, 0.234], 6/6 pairs positive | Section 3 |
| X6 E2 repair | **gate failed**: no radius passed V1-V5 on both dev checkpoints; X6 ends | Section 4 |
| X7 training replication (seeds 44, 55, 66) | pre-registered, submitted; no result | Section 5 |

What X5 shows: at the same selected initial success range on fresh 3-input problems, the online branches started from
extended BC lost more coverage (pass@16) than the branches started from light BC, for the 3 existing training seeds.
What X5 does not show: that the effect holds for new training runs (X7 tests this), why it happens, or that it holds
for other models, tasks or lengths of training.

## 1. Question

Does extended BC (128 steps) make intermittently successful problems more vulnerable to online training than light BC
(16 steps), at the same initial success probability? The question comes from a **post hoc** observation in the initial
study (its Section 8.2). That observation is not evidence here; X5 tests it on new problems with a pre-registered rule.

## 2. X1 audit of the frozen report (done; project `6d70485`; `audits/report-audit-20261026.md`)
- Test GRPO vs filtered: **4 of 18** distinct checks exclude zero, not "5 of 24" (pass^1 duplicated pass@1).
- Validation GRPO vs filtered: 1 of 18 checks excludes zero; the second "exception" has a lower limit of 0.000.
- Retention: the "no loss" statement covers 13 problems of one initial adapter (extended-BC seed 33) only.
- Band scope: bands are simultaneous over k within one branch and metric family; they resample problems with the
  adapters fixed, so they do not cover training-seed variability.

## 3. X5 result: fresh-problem confirmation

### 3.1 Design (pre-registered, protocol Section 3)
- **Panels:** xc3 = 768 fresh 3-input problems (primary), xc5 = 192 fresh 5-input problems. Generator v1, new seed
  20261026; every key in an earlier split was rejected. For 3 inputs, 200 of 968 accepted draws (20.7%) were such
  duplicates. The panel therefore under-represents the most frequent 3-input problems; X5 describes this panel.
- **Adapters:** the 6 BC adapters (light, extended x seeds 11, 22, 33) and the 12 round-256 online endpoints of the
  frozen study. Each initial adapter: 48 attempts per problem; attempts 0-23 **select** the subgroup and attempts 24-47
  give the **baseline**. Each final adapter: 32 attempts. Sampling at temperature 1, cap 256 tokens.
- **Primary statistic:** problems with 1-11 selection successes of 24, in bins 1-2 / 3-5 / 6-11. Per problem, excess
  loss e = loss_16(final) - loss_16(baseline), where loss_16 is the unbiased estimate of (1 - p)^16. Bin means use
  common weights (pooled selection shares: 0.393 / 0.364 / 0.243). Delta(m, s) = E_extended - E_light; the statistic is
  the mean of the 6 Deltas. 95% interval: 10,000 paired bootstrap resamples of problems (shared by all 18 adapters).
- **Decision:** CONFIRMED if the lower limit > 0 and >= 5 of 6 Deltas > 0; CONTRADICTED if the upper limit < 0.

### 3.2 Primary result (one look; job 10938317, `results/bc2rl-x5-an-main-s0/x5_primary.json`)
**Delta = +0.203, 95% CI [0.171, 0.234]; 6 of 6 pairs positive. Decision: CONFIRMED.**

| pair (method, seed) | E_light | E_extended | Delta | 95% CI |
|---|---:|---:|---:|---|
| GRPO, 11 | 0.115 | 0.308 | +0.193 | [0.135, 0.249] |
| GRPO, 22 | 0.107 | 0.327 | +0.221 | [0.171, 0.271] |
| GRPO, 33 | 0.276 | 0.366 | +0.090 | [0.038, 0.142] |
| filtered, 11 | 0.127 | 0.334 | +0.208 | [0.154, 0.262] |
| filtered, 22 | 0.030 | 0.368 | +0.338 | [0.293, 0.384] |
| filtered, 33 | 0.192 | 0.364 | +0.171 | [0.123, 0.219] |
| mean, GRPO | | | +0.168 | [0.133, 0.203] |
| mean, filtered | | | +0.239 | [0.206, 0.272] |

E is the standardized mean increase in the estimated probability that 16 attempts all fail. All 12 branches have E > 0:
online training lowered pass@16 on this subgroup for both initializations, and more for extended BC.
An independent recomputation from the harvested counts (laptop script, not project code) gave the same mean (0.2034)
and the same six Deltas.

### 3.3 Pre-registered sensitivity and secondary analyses
| analysis | Delta | 95% CI |
|---|---:|---|
| selection and baseline roles swapped | +0.213 | [0.184, 0.242] |
| K = 4 / 8 / 24 | +0.262 / +0.251 / +0.173 | lower limits 0.234 / 0.221 / 0.139 |
| fine bins (1, 2, ..., 11) | +0.225 | [0.192, 0.258] |
| with the 12-23 bin added | +0.203 | [0.171, 0.234] |
| consistency, change in p^16 | -0.162 | [-0.193, -0.130] |
| bins with baseline rates matched within 0.02 | not computable | no bin met the rule |

- **Consistency:** a negative Delta means that extended-BC branches gained *less* consistency than light-BC
  branches. This is the same direction as the coverage result. (The script prints the label "CONTRADICTED" for this
  row because it applies the coverage rule mechanically to a measure of opposite sign; the label has no meaning here.)
- **Delta over K** (`x5_curves.csv`, simultaneous band over K = 1-24): positive at every K: +0.210 at K = 1, a
  maximum of +0.263 at K = 5, and a minimum of +0.173 at K = 24.
- **Matched-baseline rule:** within every bin the extended adapters had higher baseline success rates than the light
  adapters (for example, bin 6-11: 0.30-0.31 against 0.16-0.23; `x5_bins.csv`). The pre-registered restriction to
  matched bins therefore had no bins. Section 3.5 gives a post hoc substitute.
- **Consistent subgroup** (c_a >= 22 of 24): 0 problems for light adapters, 1-4 for extended adapters; below the
  12-problem minimum, so not analysed.

### 3.4 Whole-panel results (descriptive; `x5_contrasts.csv`, final minus initial)
- xc3: pass@1 rises and pass@16 falls in **12 of 12** branches; the simultaneous bands exclude 0 in all 24 cases.
  pass@16 change: light -0.10 to -0.30; extended -0.21 to -0.29.
- xc5: pass@16 rises in 12 of 12 branches (point estimates); the bands exclude 0 in 10 of 12.
- This repeats the frozen study's 3-input loss / 5-input gain pattern on fresh problems.

### 3.5 Checks for artifacts (post hoc; exploratory, not pre-registered)
- **Baseline mismatch inside bins.** Stratification on exact (c_a, c_b) cells, so that the selection and baseline
  counts are both equal, gives a final loss_16 difference (extended - light) of +0.124 to +0.333 (mean +0.222, 6/6
  positive) and a final success-rate difference of -0.19 to -0.26. Extended-BC branches end lower although their
  baselines are higher, so the baseline mismatch cannot produce the sign of the result.
- **Truncation.** Truncated outputs count as failures. On xc3 they are more frequent in light-BC finals (filtered:
  8.1%, 28.7%, 25.4%; GRPO: 0.0%, 6.4%, 0.0%) than in extended-BC finals (0.1%-0.7%, except extended-filtered-s33:
  13.8%). Truncation therefore works against the observed direction.
- **Output length.** Extended-BC finals produce shorter outputs (mean 24-36 tokens, except extended-filtered-s33: 82)
  than light-BC finals (23-115). This report does not interpret the length difference.

### 3.6 Interpretation and limits
- **Observation:** on fresh 3-input problems, at the same selected initial success range, all 6 extended-light pairs
  show larger coverage loss for extended BC; the problem-level interval excludes 0 by a wide margin.
- **Interpretation:** the post hoc observation of the initial study generalizes to new problems **for these 3
  training seeds**. The interval covers problem sampling only. Seed-to-seed variation is large (Delta 0.090-0.338;
  E_light 0.030-0.276), so the result for new training runs is open. X7 tests it.
- **Not shown:** a mechanism; that the effect depends on the BC length itself and not on another difference between
  the 16-step and 128-step checkpoints (for example, the different distribution of initial success: extended adapters
  have more problems in the higher bins); generality to other models, tasks or training lengths.

## 4. X6 result: the repaired E2 assay failed its gate

### 4.1 Design (protocol Section 4; Sections 9.1-9.2)
fp32 measurement; radius grid 1/16, 1/8, 1/4, 1/2, 1 AdamW step; 2 development checkpoints (light BC and extended BC,
seed 7). A radius is valid only if V1-V5 pass on both: V1 repeat noise; V2 pair-contrast and median first-order
error <= 25%; V3 quadratic <= 20% of linear (n - 1 displacements); V4 random-direction ratio in [0.8, 1.25];
V5 median KL <= 0.01. Gate: the largest radius that passes on both. No pass: X6 ends with a documented failure.

### 4.2 Runs
- **Attempt 1** (job 10937583): failed before any check. The light-BC dev adapter solved 8.1% of 3-input attempts,
  so only 98 of 1,024 source prompts gave a mixed group; the pools were too small and 0 pairs were feasible (rule:
  >= 2). Recorded as a failed calibration (protocol 9.2). Attempt 2 doubled the source pool to 2,048 prompts and
  changed nothing else; protocol 9.2 fixed this before any V1-V5 value existed and made it the last pool change.
- **Attempt 2** (job 10937785, commit `0a916a3`, 59/59 tests passed in job 10937784): 4 feasible pairs on each
  checkpoint (218 and 438 mixed groups). Source: `results/bc2rl-x6-calib-dev-s7/{devlight,devext}/calibration.json`,
  `x6-gate.json`.

### 4.3 Result
| radius | light BC: V1-V5 | extended BC: V1-V5 |
|---|---|---|
| 1/16 | fails V4 | fails V4 |
| 1/8 | **pass** | fails V4 |
| 1/4 | **pass** | fails V4 |
| 1/2 | **pass** | fails V4 |
| 1 | fails V3 (quad/lin 0.308) | **pass** |

No radius passes on both checkpoints. **Gate: failed (`chosen_rho` = None). The E2 study run does not take place.**
No check was relaxed.

### 4.4 Observations about the failure (no change to the decision)
- V4 failed on one random direction at a time. The failing directions had a very small predicted change: devext `rand1`
  had |predicted| <= 1.5e-7 (ratios -5.68 at 1/16 and 0.498 at 1/2). The pair displacements changed the
  log-likelihood by about 1e-5 to 3e-3 (largest observed: 2.56e-3).
- V1 did not test anything: the fp32 repeat noise was exactly 0 on both checkpoints.
- V2 passed at every radius on both checkpoints (largest pair error 3.7% of the predicted contrast). The held-out
  contrast sign agreed in 4 of 4 pairs at every radius. Both are diagnostics only; they do not validate the assay.
- **Hypothesis (not tested):** a ratio criterion is unstable for directions that are almost orthogonal to the anchor
  gradient. A redesigned V4 would need a new pre-registration and new development data; this extension does not do it.

## 5. X7: training replication (pre-registered, protocol Section 9.3; submitted)
- **Why:** X5 covers 3 training seeds per initialization; new seeds can change the conclusion (Section 3.6).
- **Design:** new seeds 44, 55, 66; frozen broad-BC path (light = step 16, extended = step 128); the 4 frozen online
  branches per seed, 256 rounds; evaluation on xc3 / xc5 with the X5 protocol. Configs differ from the seed-33 configs
  only in seed, run ID, stream seed and init path (`test_x7_configs_match_frozen_protocol`).
- **Decision (one look):** REPLICATED if the X5 statistic over the 6 new pairs has a 95% lower limit > 0 **and** the
  seed mean Delta_s > 0 for all 3 new seeds; CONTRADICTED if the upper limit < 0; otherwise NOT REPLICATED.
  Secondary: a six-seed summary (11-66) with a Student-t interval (df = 5).
- **Stop rule:** X7 is the last training stage of A5. A final report follows, whatever the outcome.

## 6. Observations, interpretations, hypotheses
| type | statement | source |
|---|---|---|
| observation | X5 primary Delta +0.203 [0.171, 0.234], 6/6 positive | `x5_primary.json` |
| observation | xc3 pass@16 falls in 12/12 branches; xc5 pass@16 rises in 12/12 (10/12 bands exclude 0) | `x5_contrasts.csv` |
| observation | X6 gate failed (no common radius) | `x6-gate.json` |
| interpretation | the post hoc pattern of the initial study holds on new problems for the 3 existing seeds | Section 3.6 |
| interpretation | under the pre-registered checks, the fp32 sub-step E2 assay has no radius that is valid on both dev checkpoints | Section 4 |
| hypothesis | the effect holds for new training seeds (X7 tests it) | Section 5 |
| hypothesis | V4 fails because of near-orthogonal random directions (not tested) | Section 4.4 |

## 7. Jobs (Slurm is the authority for state)
| job | expid | resources | status at 10:57 EDT |
|---|---|---|---|
| 10932944/45/51/54/55 | bc2rl-x5-evalw1..w5 | 1 preempt GPU each | completed (w1 preempted twice) |
| 10938317 | bc2rl-x5-an-main-s0 | CPU | completed |
| 10937583, 10937713, 10937785 | bc2rl-x6-calib-dev-s7 | 1 preempt GPU | attempt 1 failed; 10937713 stopped at the test marker (0 s); attempt 2 completed |
| 10938632 | bc2rl-x7-tests-main-s0 | CPU | running |
| 10938634 | bc2rl-x7-bc-main-s0 (3 BC seeds) | 1 preempt GPU | waits for tests |
| 10938645, 10938647, 10938656, 10938662 | bc2rl-x7-w1..w4 | 1 preempt GPU each | wait for BC |
| not yet submitted | bc2rl-x7-w5 | 1 preempt GPU | after BC (the 5-slot cap counts pending jobs) |
| not yet submitted | bc2rl-x7-an-main-s0 | CPU | after all workers |

## 8. Resources and forecast
**Use (sacct, 2026-10-26 10:57 EDT).** A5 so far: **7.71 GPU-h** (X5 evaluation 6.97, including w1's two preempted
attempts; X6 0.74) and 1.74 CPU-h. Cumulative with A0-A4: 21.51 GPU-h. A5 has no GPU-hour cap.

| stage | forecast GPU-h | basis |
|---|---:|---|
| X7 BC (3 seeds) | 0.3 | 5.5 min per seed on A100 (jobs 10916268/70) |
| X7 online (12 branches) | 22.8 | 1.9 GPU-h per branch (frozen study) |
| X7 evaluation | 7.5 | X5 throughput 21.5-66.8 completions/s |
| X7 analysis | 0 | CPU |
| **X7 total** | **about 31** | |

**Timeline (estimates, EDT).** BC job: start 11:00-17:00, 20-30 min run. Workers: 6-7 h after they start, plus
preemption losses. X7 result: 2026-10-26 22:00 to 2026-10-27 12:00. Final report: within 2 h of the X7 result.
Queue measured for X5: the workers started 7.4-8.6 h after their data dependency cleared (sacct).

## 9. Blockers
None. Queue wait is the main source of delay.

## 10. Changes since update 1
- X5 evaluation and analysis completed; X5 result added (Section 3).
- X6 attempts 1 and 2 completed; X6 gate failed (Section 4); code change for insufficient pairs (`76f5150`) and
  a test-fixture fix (`0a916a3`).
- X7 pre-registered (`b65d9cd`) and submitted.
- Recipes: the GPU-memory precondition now runs (done-predicate "false"). GPU recipes now require a passing
  test-suite marker for their own commit.
