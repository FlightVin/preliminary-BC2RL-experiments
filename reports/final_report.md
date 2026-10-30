# BC2RL final report: what we know, how sure we are, and what is open

**Status: final, 2026-10-29 (owner amendments A7 and A8).** The owner stopped the study on 2026-10-29 at 18:23 EDT.
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
- [S3 PENDING: development transfer contrasts, from the read-only export.]

### 4.5 Sequence-level and prefix objectives (Stage G; development runs, seed 11)
Stage G asks whether GSPO (sequence-level ratio and clipping) or CTPO (cumulative-prefix ratio) changes the coverage
loss, beyond the effects of length normalization and of several updates per batch. Arms: C1 (reference GRPO), L1
(length-normalized, 1 update), L4 (length-normalized, 4 updates), S4 (GSPO, 4 updates), T4 and T4F (CTPO, 4 updates,
adaptive or fixed clip band).
- **Validation passed** (GSPO and CTPO equal length-normalized GRPO at the first update). BF16 training passes were too
  noisy for the sequence ratio (noise 48-78% of the signal), so all arms train with FP32 gradient passes.
- **Clip bands were calibrated before any dev run:** token band 0.011, sequence band 0.016. The GSPO paper's band
  (3e-4) would have clipped 92-95% of sequences in this setting.
- **The 11 harvested dev runs reached round 256** without failure, with 1-34 skipped rounds each (a round with no
  usable gradient takes no step). The 12th (extended T4) has also finished: its yd evaluation, which needs all 12
  run summaries, has started. Its record is not yet harvested. In-run validation pass@1 (144 problems x 4 attempts; a
  health check, not an outcome) rose in every run: light from 0.03 to 0.14-0.20, extended from 0.13 to 0.20-0.23.
  The clipped fraction of the 4-update arms was 0.07-0.27 of units per step.
- [G PENDING: yd evaluation of the 12 dev finals (g-yd-s11), then the dev comparison. Gate D is not decided.]
- The confirmation runs did not run.

## 5. What remains open, and the next step for each

| open question | why it is open | next step |
|---|---|---|
| Do harder-problem updates add to the loss? (C1) | Seed interval [-0.08, +0.10] at 6 seeds | About 20 paired seeds for the planned precision; or a design with lower seed variance |
| BC length itself vs the initial distribution it produces | 3 matched pairs; none spans step 16 vs 128; no seed interval | Build BC checkpoints designed to overlap in initial success (for example, different data at equal steps) |
| Does the objective matter? | Seed interval includes 0 (5 of 6 seeds positive) | More seeds for the GRPO vs filtered interaction (Stage G compares other objectives and does not answer this) |
| Why are multiplication-needing problems lost? | Association only; the classes also differ in target size and number of traces | S3 confirmation (at least 3 seeds) with the S2 matched pools; then a design that matches traces as well |
| Does the source of training targets change retention? | R3 dev only (one seed) | R3 confirmation as frozen in protocol 14.9.3 (about 45-55 GPU-h) |
| Do GSPO or CTPO change coverage loss? | Dev runs only; yd evaluation [G PENDING] | Gate D on the dev results, then the confirmation runs of protocol 12 |
| Does any of this generalize? | One model size, one task, one training length | Not tested; any claim beyond this setting is unsupported |

## 6. Limits on every claim
- One base model (Qwen2.5-0.5B-Instruct, LoRA r16), one synthetic arithmetic task, 256 online rounds, 6 training
  seeds at most. Nothing here tests other models, tasks or longer training.
- 10-input problems stayed near zero throughout. The study says nothing about long-horizon capability.
- Look 1 applied no correction for multiple tests. The exploratory results (Section 4) reuse panels or use one seed.
- The BC path is not bit-reproducible; re-run adapters differ from the originals by less than the seed spread.

## 7. Resources
- GPU use from job 10947200 (A6 start) to 18:53 EDT on 2026-10-29: **124.1 GPU-hours**, all on preempt GPUs
  (56 job attempts; Slurm `sacct`, BC2RL job names, all attempts including preempted ones). By stage: A6 wave 1 76.2;
  Stage G 17.3; R1 7.9; R3 13.9; S2 1.3; S3 7.5. Before A6: 40.65 GPU-h.
- The study never used general GPUs and never held more than 5 GPU slots. From 18:23 on 2026-10-29 it held one.

## 8. Records
- Protocols: `protocol/AMENDMENT-A5-2026-10-26.md`, `protocol/AMENDMENT-A6-2026-10-26.md` (A6 sections 12-14 hold
  Stage G, program S and stage R, with their dated additions).
- Detailed results: `reports/mechanism_followup.md` (look 1: Section 9; S1: 11.1; R1: 11.3). Gate DR, L* and the
  R3 dev improvements: protocol A6 14.9.3. Stage G dev records: harvested `summary.json` files, summarized by the
  workspace script `audits/scripts/g_dev_summary.py`.
- The project brief as it stood at the stop: `reports/PROJECT-A6-record-20261029.md`.
- Audits: the harness workspace `audits/` (not in this repository).
