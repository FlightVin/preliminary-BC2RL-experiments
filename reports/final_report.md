# BC2RL final report: what we know, how sure we are, and what is open

**Status: final, 2026-10-30 (owner amendments A7 and A8).** The owner stopped the study on 2026-10-29 at 18:23 EDT.
This report closes it. A fresh run starts from a reset workspace. The detailed records stay in place:
`reports/initial_study.md`, `reports/extension_study.md`, `reports/mechanism_followup.md` and the protocols in
`protocol/`. This report adds no new test. It collects the results, states how sure each one is, and lists the open
questions with the next step for each.

## 1. The answer in brief

We trained Qwen2.5-0.5B-Instruct (LoRA r16) on a verifiable arithmetic task: combine 3, 5 or 10 input numbers to
reach a target. Training had two phases: behavior cloning (BC) on solver traces, then online reinforcement learning
(GRPO or filtered supervision) for 256 rounds.

1. **Online training trades coverage for reliability.** After online training, more problems are solved on every
   attempt, but some problems that were solved only sometimes are no longer solved at all.
2. **Models that start from longer BC show a larger loss.** On fresh 3-input problems that the starting model solved in 1-11 of 24
   attempts, online training removes more coverage after 128 BC steps than after 16 BC steps. The difference is
   about +0.2 in loss_16 = (1 - p)^16. This is confirmed on fresh problems and replicated on new training seeds.
   It does not show that BC length itself is the cause (see item 5).
3. **This is not a worse final model.** Models that start from longer BC end better on the whole panel (pass@1 and
   pass@16). The larger loss is on a subgroup of problems the starting model solved only sometimes.
4. **It is not a truncation artifact, and it does not need harder problems.** Letting the same attempts run to 1,024
   tokens recovers no solution. The effect appears when updates come only from 3-input problems.
5. **It grows steadily with BC length,** from step 16 to 32, 64 and 128 (the 16-32 increase is shown on the
   problem interval only). For the few starting models we could match
   on initial competence, the initial success distribution accounts for the step effect. This match is weak (3 pairs,
   problem-level interval only).
6. **We do not know the mechanism.** Whether harder-problem updates add to the loss, whether the training objective
   matters, and why the loss falls on particular problems all remain open.

## 2. Background: how the question arose

| study | what it found | report |
|---|---|---|
| Initial study (A1-A4) | Online training raised pass@1 and pass^15 in 12 of 12 branches, while pass@15 at 3 inputs fell (point estimates; no interval). A post-hoc look suggested longer BC loses more problems. Two planned experiments (E2, E4) failed their validation gates and did not run. 10-input problems stayed near zero, so long-horizon learning was not testable. | `initial_study.md` |
| Extension (A5) | The post-hoc observation was tested on fresh problems with a pre-registered rule. **Confirmed** (Delta = +0.203, problem-bootstrap 95% CI [0.171, 0.234]; six-seed t-interval [0.141, 0.262]) and **replicated** on 3 new training seeds (+0.200, [0.168, 0.231]); 12 of 12 (method, seed) pairs positive. | `extension_study.md` |
| Mechanism follow-up (A6) | Tests of five explanations at one pre-registered look (look 1), then three exploratory programs (G, S, R). | `mechanism_followup.md` |

Delta is the extended-minus-light difference in excess loss_16 on the intermittently solved subgroup, using the
estimator of protocol A6 Section 3.4 (bins of initial success, pooled weights).

## 3. What the A6 look 1 established

Look 1 ran once on fresh confirmation panels (768 3-input problems), at commit c2f55bd. A claim needs two intervals
to exclude 0: a paired problem bootstrap (10,000 resamples) and a seed-level t-interval (6 seeds, df 5). The problem
interval holds the trained models fixed; only the seed interval covers variation between training runs. No
correction for multiple tests was applied.

| question | result | how sure |
|---|---|---|
| Does truncation at 256 tokens hide solutions? | **No.** Delta is +0.185 at both 256 and 1,024 tokens; continuation recovered no solution for any of the 36 models in this test. | Both intervals. Holds up to 1,024 tokens; longer outputs were not tested. |
| Does the effect need updates on harder (5- and 10-input) problems? | **No.** With harder-problem gradients masked, the effect remains (C2 = +0.147, 94% of Delta_M). | Both intervals. |
| Do harder-problem updates add to it? | **Unresolved.** C1 = +0.009; seed interval [-0.078, +0.095]. | The data allow anything from a small negative contribution to about 60% of the effect. About 20 seeds would be needed. |
| Does BC length matter as a dose? | **Yes, gradually.** E rises from +0.163 (step 16) to +0.208, +0.251 and +0.319 (step 128). | Both intervals for the step 32-64 and 64-128 increments; problem interval only for 16-32. |
| Is it BC length itself, or the initial success it produces? | **The initial success distribution explains it for 3 matched pairs** (mean D = -0.014, [-0.040, +0.011]). | Problem interval only. No pair spans step 16 vs 128; the seed-level spread is as large as the decision band. Weak. |
| Does the objective (GRPO vs filtered) change it? | **Not shown.** I = +0.058; seed interval [-0.008, +0.124]. | 5 of 6 seeds positive, but the seed interval includes 0. |
| Do the longer-BC models end better overall? | **Yes.** Whole-panel pass@1 +0.113 and pass@16 +0.129 for extended minus light. | Both intervals. |
| Which failure types make up the difference? | Illegal-operand failures at cap 256 (D_I = +0.328). | Both intervals at cap 256 only. Attribution is unstable: failures appear to move between the illegal-operand and wrong-target codes. |
| Which problems are lost? | Problems that need multiplication lose more (H-S1, +0.31 / +0.37). The loss correlates with how much the problem's best solution form loses favour (H-S3, r about 0.54-0.60). The init difference is not shown to depend on multiplication (H-S2). | H-S1 both intervals; H-S3 seed interval only. These are associations; H-S3 partly restates H-S1. No causal claim. |

**What this rules out and what it leaves.** Truncation (R1) is ruled out for this effect. Same-difficulty
concentration (R4) fits the main result, but its secondary prediction C4 >= 0 failed at K = 1 and at K = 4 (step 16).
A cross-difficulty contribution (R3) is not excluded, although its secondary prediction C3 > 0 failed at K = 1
(step 128). BC exposure beyond the initial distribution (R5) is not shown;
the initial-distribution account (R6) fits weakly. A method effect (R7) is not shown.

## 4. Exploratory results after look 1

None of these results is a confirmed claim. Each comes from development data, one seed, or reused panels.

### 4.1 Cross-difficulty contrasts per starting model (S1; reused look-1 panel)
No harder-update contrast has both intervals above 0 for either starting model. The data do not show that 5- and
10-input updates harm 3-input coverage. They also cannot exclude an effect of about +0.06 per starting model. Two
light-BC contrasts are positive on both intervals (equal total samples M - T3: +0.027; equal 3-input exposure
MASK - T3E: +0.045), but each comparison differs in more than one way, so neither isolates a cause.

### 4.2 Policy drift (R1; existing checkpoints, dev panel)
- The policy moves further from its starting point as training goes on, in all 10 branch types. At round 256,
  light-BC branches move further than extended-BC branches.
- Most per-problem associations with drift are driven by response length. The sequence KL is a sum over tokens, so
  longer responses show more drift. After controlling for length, drift has no consistent association with excess
  loss.
- Two length-controlled associations are consistent across seeds: drift rises with improvement for light BC (5 of 6
  seeds), and falls with initial competence for extended BC (0 of 6 seeds positive).
- Adding drift to a prediction of a branch's loss gives an inconclusive change in error (12 branches; RMSE 0.072
  without drift, 0.077 with sequence drift, 0.063 with per-token drift). These are hypotheses only.

### 4.3 Response-source experiment (R3; development block only)
R3 asks whether retention depends on where the supervised training targets come from: the solver trace (S), one
fixed response of the starting model (I), a fresh response of the frozen starting model (F), or a fresh response of
the current model (O).
- **Gate DR passed** on the dev seed: pool 256 problems, 256 rounds per arm, masked fraction 0.044 (light) and 0.019
  (extended) against a limit of 0.05, all losses finite.
- Dev improvement in yd3 pass@1 at round 256: light S +0.26, I +0.25, F +0.28, O +0.37; extended S +0.06, I +0.11,
  F +0.12, O +0.17. The current-policy arm (O) improved most in both starting models. One seed; no interval.
- The matched-progress level was fixed at L* = 0.07. Its checkpoint rule finds a checkpoint for only 2 of 8 dev
  branches, because most branches are already above 0.08 at round 32, the first saved round. Matched-progress comparisons would
  therefore be mostly unavailable.
- **The confirmation runs (6 blocks, 24 branches; protocol 14.9.3) did not run.** The owner stopped the study before
  they started.

### 4.4 Classes with and without multiplication (S2, S3)
- **S2 built a matched dataset** (650 matched pairs; gate S2 passed with 256 problems per class). The classes are
  still not equal: class A targets are smaller (mean 14.0 vs 36.2) and class A problems have more solution traces
  (4.08 vs 1.04), and class A has higher prescreen success (light 0.086 vs 0.072; extended 0.254 vs 0.190). A difference between the classes is therefore not attributable to multiplication alone.
- **S3 development runs are complete** (seed 11; 5 arms x 2 starting models; all 12 worklist tasks done by
  2026-10-29 16:46, workspace log). They test whether training on one class
  helps or harms the other.
- **S3 asks whether updates on one class change coverage on the other.** The contrast T_B->A compares AB with AB-mB:
  both arms see the same prompts, but AB-mB takes no gradient from class B. T_B->A > 0 means the class B updates
  add coverage loss on class A. T_A->B is the same test in the other direction. E is the standardized excess loss_16
  on the class subgroup (1-11 of 24 initial successes) of the 3-input dev panel (498 problems: A 262, B 236).
  Intervals are problem-bootstrap 95% only, because there is one seed.

  | contrast | light BC | extended BC | extended - light |
  |---|---|---|---|
  | T_B->A (B updates, effect on A) | **+0.226** [+0.157, +0.291] | +0.018 [-0.014, +0.049] | [-0.277, -0.131] |
  | T_A->B (A updates, effect on B) | -0.038 [-0.098, +0.019] | +0.002 [-0.072, +0.077] | [-0.040, +0.117] |

- **From light BC, class B updates add coverage loss on class A.** The interval excludes 0. From extended BC, the
  effect is not distinguishable from 0, and the init difference excludes 0. Class A updates show no effect on class B
  from either starting model.
- **Single-class training shows the same direction more strongly.** Training on B only gives E on class A of +0.83
  (light) and +0.87 (extended), and class A pass@1 falls by 0.10 and 0.33. Training on A only gives E on class B of
  +0.47 and +0.61, with pass@1 almost unchanged (-0.01, -0.08). These single-class arms have no masked control.
- **Class B gave less learning signal than class A.** In the AB arm, the share of mixed (informative) groups was 0.15
  (light) and 0.21 (extended) for B, against 0.35 and 0.37 for A. Equal prompts did not give equal signal.
- **These results are exploratory.** They come from one seed, they have no seed interval, and the S3 confirmation
  (at least 3 seeds) did not run. The S2 confounds above apply: a class effect is not attributable to multiplication
  alone. The source is the workspace audit script `audits/scripts/s3_dev_analysis.py` on the S3 export (job 10988302).

### 4.5 Sequence-level and prefix objectives (Stage G; development runs, seed 11)
Stage G asks whether GSPO (sequence-level ratio and clipping) or CTPO (cumulative-prefix ratio) changes the coverage
loss, beyond the effects of length normalization and of several updates per batch. Arms: C1 (reference GRPO), L1
(length-normalized, 1 update), L4 (length-normalized, 4 updates), S4 (GSPO, 4 updates), T4 and T4F (CTPO, 4 updates,
adaptive or fixed clip band).
- **Validation passed** (GSPO and CTPO equal length-normalized GRPO at the first update). BF16 training passes were too
  noisy for the sequence ratio (noise 48-78% of the signal), so all arms train with FP32 gradient passes.
- **Clip bands were calibrated before any dev run:** token band 0.011, sequence band 0.016. The GSPO paper's band
  (3e-4) would have clipped 92-95% of sequences in this setting.
- **All 12 dev runs reached round 256** with finite losses, with 1-34 skipped rounds each (a round with no usable
  gradient takes no step). In the 4-update arms, the first step of every round had a ratio of exactly 1, as designed.
  The mean clipped fraction over all steps was 0.07-0.27 of units.
- **Gate D (decided on training diagnostics only) is not met in one run.** The rule needs the step-4 median token
  |log ratio| of L4 to be at least 3e-4. Light L4 reaches 3.0e-3; extended L4 reaches only 1.0e-4. S4 (sequence
  median 0.010-0.011) and the clipping fractions (0.10-0.37 over steps 2-4) pass. By the protocol rule, a D2 failure
  stops Stage G and the report states that the experiment does not exercise the distinction for that arm. Two parts
  of the gate were not checked from the laptop: the 3x noise-floor condition and the CTPO cumulative-ratio median.
  This reading comes from the workspace audit script `audits/scripts/g_gate_d.py`, not from a project gate job.
- **yd results of the 12 finals (descriptive; one seed, no interval; 384 3-input and 96 5-input problems, 32
  attempts each):**

  | init | panel | C1 | L1 | L4 | S4 | T4 | T4F |
  |---|---|---|---|---|---|---|---|
  | light | yd3 pass@1 / pass@4 | 0.341 / 0.452 | 0.359 / 0.548 | 0.376 / 0.411 | 0.393 / 0.458 | 0.377 / 0.419 | 0.367 / 0.512 |
  | extended | yd3 pass@1 / pass@4 | 0.488 / 0.587 | 0.475 / 0.624 | 0.506 / 0.560 | 0.492 / 0.590 | 0.513 / 0.578 | 0.512 / 0.617 |
  | light | yd5 pass@1 | 0.000 | 0.062 | 0.088 | 0.000 | 0.050 | 0.004 |
  | extended | yd5 pass@1 | 0.095 | 0.068 | 0.150 | 0.000 | 0.100 | 0.090 |

  - On 3-input problems, all objectives end within about 0.05 pass@1 of each other in each starting model. The
    pre-registered contrasts (L1 - C1, L4 - L1, S4 - L4) are between -0.014 and +0.031 in pass@1.
  - Four updates per batch (L4) raise pass@1 a little over one update (L1) but lower pass@4 (light 0.548 to 0.411;
    extended 0.624 to 0.560). This is the same reliability-for-coverage trade seen in the main study, here from
    rollout reuse. One seed; it needs confirmation.
  - **GSPO (S4) ends at 0 on the 5-input panel for both starting models.** From extended BC, every other arm keeps
    0.068-0.150; from light BC, C1 also ends at 0 and T4F at 0.004. This is the largest difference in the table. It
    is one seed, and its cause is not examined.
  - The coverage-loss estimator (E) was not computed for the dev finals: the per-problem counts stay on compute
    storage, and the stop came first.
- The confirmation runs did not run.

## 5. What remains open, and the next step for each

| open question | why it is open | next step |
|---|---|---|
| Do harder-problem updates add to the loss? (C1) | Seed interval [-0.08, +0.10] at 6 seeds | About 20 paired seeds for the planned precision; or a design with lower seed variance |
| BC length itself vs the initial distribution it produces | 3 matched pairs; none spans step 16 vs 128; no seed interval | Build BC checkpoints designed to overlap in initial success (for example, different data at equal steps) |
| Does the objective matter? | Seed interval includes 0 (5 of 6 seeds positive) | More seeds for the GRPO vs filtered interaction (Stage G compares other objectives and does not answer this) |
| Why are multiplication-needing problems lost? Do class B updates harm class A? | Association only; the classes also differ in target size and number of traces. S3 dev (one seed): class B updates add loss on class A from light BC only | S3 confirmation (at least 3 seeds) with the S2 matched pools; then a design that matches traces as well |
| Does the source of training targets change retention? | R3 dev only (one seed) | R3 confirmation as frozen in protocol 14.9.3 (about 45-55 GPU-h) |
| Do GSPO or CTPO change coverage loss? | Dev runs only (one seed); gate D not met for extended L4 (workspace audit script); E not computed | Check gate D in a project job (with the noise floors); if the owner wants Stage G to continue, decide how to handle the D2 failure in a dated addition, then confirmation runs. Examine the GSPO 5-input collapse first |
| Does any of this generalize? | One model size, one task, one training length | Not tested; any claim beyond this setting is unsupported |

## 6. Limits on every claim
- One base model (Qwen2.5-0.5B-Instruct, LoRA r16), one synthetic arithmetic task, 256 online rounds, 6 training
  seeds at most. Nothing here tests other models, tasks or longer training.
- 10-input problems stayed near zero throughout. The study says nothing about long-horizon capability.
- Look 1 applied no correction for multiple tests. The exploratory results (Section 4) reuse panels or use one seed.
- The BC path is not bit-reproducible; re-run adapters differ from the originals by less than the seed spread.

## 7. Resources
- GPU use from job 10947200 (A6 start) to the last GPU job (g-dev4, ended 2026-10-30): **133.2 GPU-hours**, all on
  preempt GPUs (60 job attempts; Slurm `sacct` read 2026-10-30, BC2RL job names, all attempts including preempted
  ones). By stage: A6 wave 1 76.2; Stage G 26.4; R1 7.9; R3 13.9; S2 1.3; S3 7.5. Before A6: 40.65 GPU-h.
- After the A8 stop, g-dev4 finished the Stage G dev runs and the yd evaluations on its one GPU. One CPU job
  (10988302) ran the tests (122 passed) and the S3 export.
- The study never used general GPUs and never held more than 5 GPU slots. From 18:23 on 2026-10-29 it held one.

## 8. Records
- Protocols: `protocol/AMENDMENT-A5-2026-10-26.md`, `protocol/AMENDMENT-A6-2026-10-26.md` (A6 sections 12-14 hold
  Stage G, program S and stage R, with their dated additions).
- Detailed results: `reports/mechanism_followup.md` (look 1: Section 9; S1: 11.1; R1: 11.3). Gate DR, L* and the
  R3 dev improvements: protocol A6 14.9.3. Stage G dev records: harvested `summary.json` files, summarized by the
  workspace scripts `audits/scripts/g_dev_summary.py` and `audits/scripts/g_gate_d.py`. S3 dev contrasts: workspace
  script `audits/scripts/s3_dev_analysis.py` on the export of job 10988302.
- The project brief as it stood at the stop: `reports/PROJECT-A6-record-20261029.md`.
- Audits: the harness workspace `audits/` (not in this repository).
