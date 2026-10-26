# BC2RL extension study (amendment A5)

**Status: FINAL report, 2026-10-26 18:30 EDT (22:30 UTC).** All pre-registered stages are complete and the A5 stop
rule applies (protocol Section 9.3): X7 was the last training stage. X5 is **CONFIRMED** on fresh problems. X6 (E2
repair) **failed** its pre-registered gate. X7 is **REPLICATED** on 3 new training seeds. Earlier updates: 0 (2026-10-25
23:49 EDT), 1 (2026-10-26 08:20), 2 (11:10), 3 (16:20).

Protocol: `protocol/AMENDMENT-A5-2026-10-26.md` (X5 pre-registration `b2ad826`; X6 `eb19ea8`, Sections 9.1-9.2;
X7 Section 9.3, `b65d9cd`). The initial study stays frozen (`reports/initial_study.md`; corrections in its Section 12).
Audit of this report: `audits/extension-audit-20261026T2228Z.md` in the harness workspace.

## 0. Summary

| stage | result | evidence |
|---|---|---|
| X1 audit | 4 count/scope errors corrected in the frozen report; no number changed | Section 2 |
| X5 confirmation (fresh problems, seeds 11, 22, 33) | **CONFIRMED**: Delta = +0.203, 95% CI [0.171, 0.234], 6/6 pairs positive | Section 3 |
| X6 E2 repair | **gate failed**: no radius passed V1-V5 on both dev checkpoints; X6 ends | Section 4 |
| X7 training replication (new seeds 44, 55, 66) | **REPLICATED**: Delta = +0.200, 95% CI [0.168, 0.231], 6/6 pairs positive; Delta_s > 0 for 3/3 seeds | Section 5 |
| six-seed summary (11-66, secondary) | mean seed Delta +0.202, t95 [0.141, 0.262], 6/6 seeds positive | Section 5.3 |

**What the extension shows.** On fresh 3-input problems that the initial adapter solved intermittently (1-11 of 24
selection attempts), online training removed more coverage (pass@16) when it started from extended BC (128 steps)
than when it started from light BC (16 steps). This holds for the 3 original training seeds (X5) and for 3 new
training seeds (X7). It holds for both online methods (GRPO and filtered) and in all 12 (method, seed) pairs.

**What it does not show.** A mechanism. That the BC length itself causes the effect, and not another difference
between the step-16 and step-128 checkpoints (for example, the different distribution of initial success; Section 6).
Generality to other models, tasks, BC lengths or online training lengths. Six training seeds describe training
variability for this setup only. E2 (retention under small updates) has no valid assay (X6).

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
  E_light 0.030-0.276), so the result for new training runs was open at that point. X7 tested it (Section 5).
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
- V4 failed on one random direction at a time. The failing directions had a small predicted change: devext `rand1`
  had |predicted| <= 1.5e-7 (ratios -5.68 at 1/16 and 0.498 at 1/2). The pair displacements changed the
  log-likelihood by about 1e-5 to 3e-3 (largest observed: 2.56e-3).
- V1 did not test anything: the fp32 repeat noise was exactly 0 on both checkpoints.
- V2 passed at every radius on both checkpoints (largest pair error 3.7% of the predicted contrast). The held-out
  contrast sign agreed in 4 of 4 pairs at every radius. Both are diagnostics only; they do not validate the assay.
- **Hypothesis (not tested):** a ratio criterion is unstable for directions that are almost orthogonal to the anchor
  gradient. A redesigned V4 would need a new pre-registration and new development data; this extension does not do it.

## 5. X7 result: training replication on new seeds

### 5.1 Design (pre-registered, protocol Section 9.3)
- **Why:** X5 covers 3 training seeds per initialization; new seeds could change the conclusion (Section 3.6).
- **Design:** new seeds 44, 55, 66; frozen broad-BC path (light = step 16, extended = step 128); the 4 frozen online
  branches per seed, 256 rounds; evaluation on xc3 / xc5 with the X5 protocol (48 / 32 attempts, seed 5, cap 256).
  Configs differ from the seed-33 configs only in seed, run ID, stream seed and init path
  (`test_x7_configs_match_frozen_protocol`, 61/61 tests passed on `b65d9cd` and `d09a5b2`).
- **Decision (one look):** REPLICATED if the X5 statistic over the 6 new pairs has a 95% lower limit > 0 **and** the
  seed mean Delta_s > 0 for all 3 new seeds; CONTRADICTED if the upper limit < 0; otherwise NOT REPLICATED.

### 5.2 Execution and acceptance
- BC (job 10938634, 1,147 s): validation curves in the range of seeds 11-33, except light seed 66 (step-16 pass@1
  0.012 against 0.031-0.040).
- 5 workers ran 15 tasks; every training and evaluation step exited 0. 7 worker preemptions; each resumed from its
  last checkpoint. One worker (job 10938999) stopped at the test-marker precondition and ran again after a test job.
- Structural acceptance before the analysis (`audits/x7_structural_accept.py`): 36/36 (18 adapters x 2 panels) passed;
  768 / 192 rows; n = 32 / 48; c_a + c_b = c; adapter path = config; 18 distinct sha256. No outcome was examined
  before the analysis job.

### 5.3 Primary result (one look; job 10943976, `results/bc2rl-x7-an-main-s0/x7_primary.json`)
**Delta = +0.200, 95% CI [0.168, 0.231]; 6 of 6 pairs positive; Delta_s > 0 for all 3 seeds. Decision: REPLICATED.**

| pair (method, seed) | E_light | E_extended | Delta | 95% CI |
|---|---:|---:|---:|---|
| GRPO, 44 | 0.208 | 0.340 | +0.132 | [0.084, 0.179] |
| GRPO, 55 | 0.115 | 0.295 | +0.181 | [0.134, 0.227] |
| GRPO, 66 | 0.099 | 0.261 | +0.162 | [0.085, 0.240] |
| filtered, 44 | 0.163 | 0.309 | +0.146 | [0.101, 0.191] |
| filtered, 55 | 0.093 | 0.379 | +0.286 | [0.239, 0.334] |
| filtered, 66 | 0.019 | 0.311 | +0.292 | [0.236, 0.359] |
| mean, GRPO | | | +0.158 | [0.121, 0.195] |
| mean, filtered | | | +0.242 | [0.210, 0.273] |

Pooled bin weights: 0.390 / 0.350 / 0.260. An independent recomputation from the harvested counts (laptop script
`audits/x7_independent_recompute.py`, not project code) gave the same six Deltas, mean (0.1999) and seed values; its own
2,000-resample bootstrap gave [0.168, 0.234].

**Seed level and six-seed summary (secondary, descriptive).**

| seed | 11 | 22 | 33 | 44 | 55 | 66 |
|---|---:|---:|---:|---:|---:|---:|
| Delta_s (mean of GRPO and filtered) | +0.200 | +0.279 | +0.131 | +0.139 | +0.234 | +0.227 |

Mean +0.202; Student-t 95% interval (df = 5) [0.141, 0.262]; 6 of 6 seeds positive. Seeds 11-33 use the X5 adapters
and evaluations; the interval treats the 6 seeds as independent training runs of one fixed setup.

### 5.4 Pre-registered sensitivity and secondary analyses
| analysis | Delta | 95% CI |
|---|---:|---|
| selection and baseline roles swapped | +0.190 | [0.160, 0.220] |
| K = 4 / 8 / 24 | +0.256 / +0.245 / +0.176 | lower limits 0.226 / 0.215 / 0.143 |
| consistency, change in p^16 | -0.186 | [-0.220, -0.151] |

- **Consistency:** a negative value means extended-BC branches gained *less* consistency than light-BC branches; this
  is the same direction as the coverage result. The script's label "CONTRADICTED" for this row is mechanical (as X5).
- **Per-bin table** (`x7_bins.csv`): in every bin and seed, the extended adapters had higher baseline success rates
  than the light adapters (bin 6-11: 0.30-0.32 against 0.13-0.23). Light seed 66 had only 12 problems in bin 6-11.
- **Whole panel** (`x7_contrasts.csv`, final minus initial): xc3 pass@1 rises and pass@16 falls in **12 of 12**
  branches; the simultaneous bands exclude 0 in 12 of 12 for both. pass@16 change: light -0.04 to -0.22; extended
  -0.20 to -0.26. xc5 pass@16 rises in 12 of 12 branches (bands exclude 0 in 10 of 12). Same pattern as X5.

### 5.5 Checks for artifacts (post hoc; exploratory, not pre-registered)
- **Baseline mismatch inside bins.** Exact (c_a, c_b)-cell matching (`audits/x7_posthoc_cellmatch.py`, the X5 method)
  gives a final loss_16 difference (extended - light) of +0.148 to +0.326 (mean +0.221, 6/6 positive) and a final
  success-rate difference of -0.19 to -0.30. As in X5, extended-BC branches end lower although their baselines are
  higher.
- **Truncation** (counted as failure; xc3 finals). Unlike X5, truncation is not one-sided: extended-grpo-s44 truncated
  31.2% of attempts (other extended finals 0.0-6.9%); light finals 0.0-21.0%. grpo-s44 has the smallest pair Delta
  (+0.132), so truncation in extended-grpo-s44 can contribute to that pair. The other 5 pairs use extended finals
  with 0.0-6.9% truncation. Seed 44 stays positive on its filtered pair alone (+0.146). This report does not correct
  for truncation, because the protocol does not.
- **Output length.** Mean generated tokens on xc3: extended finals 21-112, light finals 44-99. Not interpreted.

## 6. Interpretation and limits (all stages)
- **Observation:** on fresh 3-input problems, at the same selected initial success range, extended-BC branches lost
  more pass@16 than light-BC branches in 12 of 12 (method, seed) pairs over 6 training seeds. The problem-level
  intervals (X5, X7) and the seed-level t-interval all exclude 0.
- **Interpretation:** the scope is this model (Qwen2.5-0.5B-Instruct, LoRA r16), task, BC path and 256-round online
  training. In this scope, online training from the step-128 BC checkpoint lowers pass@16 more on intermittently
  solved problems than online training from the step-16 checkpoint. The result holds over training seeds, not only over problems.
- **Confound not removed:** the two initializations differ in more than BC length as such. Within each bin the
  extended adapters have higher baseline success (Sections 3.3, 5.4). The post hoc cell-matched checks (Sections 3.5,
  5.5) keep the sign, but they are exploratory. The comparison is between two checkpoints of one BC path; it does not
  separate "more BC steps" from other properties of those checkpoints.
- **Not shown:** a mechanism; generality to other models, tasks, BC lengths or online lengths; any retention claim
  (E2 has no valid assay).

## 7. Observations, interpretations, hypotheses
| type | statement | source |
|---|---|---|
| observation | X5 primary Delta +0.203 [0.171, 0.234], 6/6 positive | `x5_primary.json` |
| observation | X7 primary Delta +0.200 [0.168, 0.231], 6/6 positive, Delta_s > 0 for seeds 44, 55, 66 | `x7_primary.json` |
| observation | six-seed mean Delta_s +0.202, t95 [0.141, 0.262], 6/6 positive | `x7_primary.json` (`six_seed`) |
| observation | xc3 pass@16 falls in 12/12 branches in X5 and in X7; xc5 pass@16 rises in 12/12 in both | `x5_contrasts.csv`, `x7_contrasts.csv` |
| observation | X6 gate failed (no common radius) | `x6-gate.json` |
| observation | extended baselines exceed light baselines in every bin (X5, X7) | `x5_bins.csv`, `x7_bins.csv` |
| interpretation | the step-128 initialization loses more coverage on intermittently solved problems than the step-16 initialization, over problems and training seeds, in this setup | Section 6 |
| interpretation | under the pre-registered checks, the fp32 sub-step E2 assay has no radius that is valid on both dev checkpoints | Section 4 |
| hypothesis | the effect depends on BC length as such, not on other checkpoint differences (not tested) | Section 6 |
| hypothesis | V4 fails because of near-orthogonal random directions (not tested) | Section 4.4 |

## 8. Jobs (Slurm is the authority; all A5 jobs ended)
| job | expid | resources | outcome |
|---|---|---|---|
| 10932942, 10933044 | bc2rl-x5-data-main-s0 | CPU | first failed; second completed |
| 10932944/45/51/54/55 | bc2rl-x5-evalw1..w5 | 1 preempt GPU each | completed (w1 preempted twice) |
| 10938317 | bc2rl-x5-an-main-s0 | CPU | completed |
| 10937583, 10937713, 10937785 | bc2rl-x6-calib-dev-s7 | 1 preempt GPU | attempt 1 failed; 10937713 stopped at the test marker (0 s); attempt 2 completed |
| 10938632, 10940400 | bc2rl-x7-tests-main-s0 | CPU | completed (61/61 each) |
| 10938634 | bc2rl-x7-bc-main-s0 | 1 preempt GPU | completed |
| 10938645, 10938647, 10938656, 10938662 | bc2rl-x7-w1..w4 | 1 preempt GPU each | completed (preemptions: w1 3, w3 1, w4 2) |
| 10938999, 10940402 | bc2rl-x7-w5 | 1 preempt GPU | 10938999 stopped at the test marker (1 s); 10940402 completed (preempted once) |
| 10943976 | bc2rl-x7-an-main-s0 | CPU | completed (28 s) |

## 9. Resources
**Use (sacct, 2026-10-26 18:27 EDT; all attempts, including preempted ones).** A5: **26.85 GPU-h** (X5 6.97; X6 0.74;
X7 19.14) and 2.27 CPU core-h (X5 1.10; X6 0.51; X7 0.67). Cumulative with A0-A4: 40.65 GPU-h. A5 had no GPU-hour
cap. All GPU work ran on the preempt partition; BC2RL never held more than 5 GPU slots, counting pending jobs
(`tools/bc2rl_guard.py`). X7 used 19.14 GPU-h against a forecast of 31: online training took 0.6-1.3 GPU-h per
branch, not the forecast 1.9.

## 10. Blockers and next experiments
No blockers. No further experiment runs under A5 (stop rule, protocol 9.3). Experiments that could change the
interpretation need a new owner amendment and pre-registration:
1. Separate BC length from checkpoint properties: initializations matched on initial success distribution (for
   example, BC paths of different lengths that reach the same validation pass@1).
2. Other BC lengths (for example, steps 32 and 64) to test a dose-response relation.
3. A redesigned E2 assay (new V4 criterion, new development data).

## 11. Changes since update 3
- X7 completed: all 18 adapters evaluated and accepted; primary REPLICATED (Section 5); independent recomputation and
  post hoc checks added.
- Summary, interpretation (Section 6), observation table, jobs and resources made final.

## 12. Changes from update 2 to update 3
- X7 tests, BC and 11 of 12 online branches completed; progress table added.
- Worker w5 stopped at the test-marker precondition and ran again after a test job on its commit; no code change.
- Resource use, X7 forecast and timeline revised; CPU-h for X5 + X6 corrected from 1.74 to 1.61.

## 13. Changes from update 1 to update 2
- X5 evaluation and analysis completed; X5 result added (Section 3).
- X6 attempts 1 and 2 completed; X6 gate failed (Section 4); code change for insufficient pairs (`76f5150`) and
  a test-fixture fix (`0a916a3`).
- X7 pre-registered (`b65d9cd`) and submitted.
- Recipes: the GPU-memory precondition now runs (done-predicate "false"). GPU recipes now require a passing
  test-suite marker for their own commit.
