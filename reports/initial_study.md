# BC2RL: signal, transfer and retention after behavior cloning — initial study

**Status: DRAFT (2026-10-25 10:30 EDT).** Sections marked **PENDING** wait for runs that are queued.
Test results are blinded until the result set is frozen (latest Tue 2026-10-27 15:00 EDT). Every number
below comes from development or validation data unless a section says "test".

## 1. Summary

This study tests three hypotheses with Qwen2.5-0.5B-Instruct (LoRA r16) on a verifiable arithmetic task with
3, 5 and 10 inputs (2, 4 and 9 operations):

| hypothesis | experiment | status |
|---|---|---|
| H1: signal availability does not fully determine transfer | E2 matched-update pairs at fixed BC checkpoints | **E2 not run**: pre-launch check failed (Section 6); E0 + observational E3 only |
| H2: aggregate improvement can conceal loss of existing competence | E3 retention on initially consistent problems | PENDING |
| H3: directional constraints preserve competence beyond shrinking | E4 projection vs norm-matched vs sham | conditional (gate in Section 8) |

Facts established so far:
- The untouched model is at the floor (dev pass@1 2.2% / 0% / 0% for 3 / 5 / 10 inputs).
- Behavior cloning (BC) creates distinct initial profiles at 3 and 5 inputs. 10 inputs stays at or near zero
  for every checkpoint measured. The study therefore cannot test long-horizon (9-operation) learning.
- Online GRPO from the light-BC checkpoint raises validation pass@1 from 0.042 to 0.184 in 256 rounds
  (seed 11), almost all of it at 3 inputs.

## 2. What changed (protocol amendment A2)

The first study (frozen 2026-10-24, commit `5deefef`) asked which initialization and online method performs
best. Amendment A2 (2026-10-25, `protocol/AMENDMENT-A2-2026-10-25.md`) changes the emphasis to the mechanism
question: *when initial policies give comparable success/failure signals, what decides whether an online
update transfers to other problems or damages existing competence?* The amendment keeps all valid runs,
the frozen task, data, model, decoding, loss and round budget. It removes the no-BC online arms (none ran) and
adds E0, E2, the retention analysis, 15-attempt validation panels and the full pass@k / pass^k grid
(k = 1..15). Section 10 lists every deviation.

Evidence classes used in this report:
- **Established mathematics:** estimators, mixed-group probability, the GRPO group coefficient, the E0 counterexample.
- **Observational evidence:** E3 trajectories and retention transitions.
- **Intervention evidence:** E2 (and E4 if run).

The existence of intermediate-probability learning signals and of gradient interference is prior work [5, 7, 8, 10, 11],
not a finding of this study.

## 3. Setup

| item | value |
|---|---|
| model | Qwen2.5-0.5B-Instruct @ `7ae5576`, BF16; LoRA r16 / alpha 32 / dropout 0 on q,k,v,o,gate,up,down (8,798,208 parameters) |
| task | n positive integers 2-20 and a target <= 1000; exactly n-1 lines `a op b = c` (+, -, *); exact verifier; reward 1 only for full success |
| data (v1) | dev 144, devdemo 512, BC 2,048, online 4,096, validation 144, test 288 (balanced by input count); mechanism splits anchor-fit / anchor-check / probe 48 each (new draws, same generator) |
| BC | AdamW 1e-4, batch 32, stratified order; light = 16 steps (0.25 epoch), extended = 128 steps (2 epochs); seeds 11, 22, 33 |
| online | 32 prompts x G = 4 completions per round, 256 rounds, one AdamW step per round (lr 1e-5), fresh optimizer at the fork; GRPO (group-standardized, clip 0.2, KL 0, ratio 1 by construction) or filtered supervision (successful fresh outputs only) |
| decoding | temperature 1, top-p 1, top-k off, repetition penalty 1, cap 256 tokens; one fixed format example in every prompt |
| evaluation | test 288 x 15 attempts; validation 144 x 4 (curves) and 144 x 15 (initial and round-256 panels) |
| compute | Babel `preempt` GPUs only, at most 5 concurrent BC2RL GPUs |

## 4. Task suitability (dev and validation; Section A of the plan)

### 4.1 Untouched model, dev, 15 attempts, format prompt (job 10915835)

| inputs | pass@1 | pass@4 | pass^4 | format valid | truncated | problems with >= 1 success |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 0.022 | 0.077 | 0 | 0.722 | 0.042 | 9/48 |
| 5 | 0 | 0 | 0 | 0.713 | 0.043 | 0/48 |
| 10 | 0 | 0 | 0 | 0.579 | 0.110 | 0/48 |

The dominant failure is `illegal_operand` (an unavailable number or a reused input). An independent re-scorer
agrees with the verifier on 6,480 / 6,480 baseline attempts.

### 4.2 Learnability (disposable dev BC path, seed 7, dev 15 attempts)

| BC steps | pass@1 3 / 5 / 10 inputs | 5-input problems solved |
|---:|---|---:|
| 0 | 0.022 / 0 / 0 | 0/48 |
| 16 | 0.097 / 0.004 / 0 | 3/48 |
| 128 | 0.307 / 0.028 / 0 | 12/48 |

The pre-registered gate (5-input pass@1 >= 2% with >= 5/48 solved) passed. The generator was not revised.

### 4.3 Main BC checkpoints, validation 144 x 4 attempts

| seed | light (16) | extended (128) | continued BC (256) |
|---|---:|---:|---:|
| 11 | 0.031 | 0.118 | 0.139 |
| 22 | 0.038 | 0.097 | 0.158 |
| 33 | 0.040 | 0.151 | 0.141 |

10-input validation pass@1 is <= 0.005 for every checkpoint.

### 4.4 Audit by input count (validation 144 x 4; range over seeds 11 / 22 / 33)

| inputs | quantity | light BC | extended BC | continued BC |
|---:|---|---|---|---|
| 3 | pass@1 | 0.089-0.115 | 0.271-0.411 | 0.380-0.422 |
| 3 | problems with a mixed 4-attempt group | 0.31-0.40 | 0.63-0.67 | 0.60-0.73 |
| 5 | pass@1 | 0-0.010 | 0.010-0.042 | 0.036-0.047 |
| 5 | problems with a mixed 4-attempt group | 0-0.04 | 0.04-0.15 | 0.13-0.19 |
| 10 | pass@1 | 0 | 0 | 0-0.005 |
| 10 | problems with a mixed 4-attempt group | 0 | 0 | 0-0.02 |

- The dominant light-BC failure is an illegal operand (an unavailable or reused number): 0.54-0.89 of attempts.
  After extended or continued BC, the dominant failure at 3 and 5 inputs is a legal chain with a wrong final value.
- Format validity is >= 0.75 for every BC checkpoint (untouched model: 0.57-0.72).
- Light BC truncates often at 10 inputs (256-token cap): 0.03, 0.29 and 0.67 of attempts for seeds 22, 11 and 33.
  Some 10-input failures of light BC are therefore length failures, not only arithmetic failures.
- Consequence: GRPO receives almost all of its signal from 3-input problems, some from 5-input problems and none from
  10-input problems. The study cannot test long-horizon learning. Source: `audit.csv` (analysis code, commit 26eae27).

![pass@k by input count, validation 4-attempt panels](figures/curves_val4_passat.png)

## 5. E0: exact two-task model (analytic control)

E0 is mathematics, not a measurement. Two tasks share one parameter theta. In the *aligned* model both tasks
improve together; in the *opposing* model one task improves while the other degrades. At theta = 0 both models have
identical success curves and identical mixed-group probabilities, but the full-batch policy gradient is 0.25 in the
aligned model and exactly 0 in the opposing model. In the opposing model, mean success (pass@1) stays at 0.5 for
every theta, while pass@k falls and pass^k rises for every k > 1. A success profile alone therefore cannot determine
whether an update transfers, and pass@1 alone can hide a change in competence. All numeric checks pass (CPU job
10921829): estimator bias <= 1e-15, GRPO and mixed-group identities exact to 2e-16, Taylor residual ratio <= 0.045.

![E0 exact two-task model](figures/e0_toy.png)

## 6. E2: matched-update intervention (H1) — not run

**E2 was not run as an intervention.** Its pre-registered pre-launch check failed at the calibrated radius, and the
decision rule fixed before the diagnostic (amendment Section 11) excludes the experiment. This is a
measurement-limited result. The design is unchanged (amendment Section 4): matched high- vs low-alignment GRPO
minibatches, equal-norm actual AdamW displacements, one radius, and the paired anchor-check log-likelihood change.

What happened (development data only; dev light-BC checkpoint, seed 7):
- The radius rule chose rho = 1, which is one real AdamW step (median sampled-token KL 0.0048; 0.021 at rho = 2).
- The Taylor check compares the actual anchor-fit log-likelihood change with the first-order prediction h.d. It needs
  a ratio in [0.5, 2]. In the BF16 model the ratios were 0.20, 4.45, 0.24 and 13.7.
- A diagnostic (job 10925931) separated three causes:
  - **Implementation:** correct. In fp32, four random directions of equal norm follow h.d with ratio 0.96-1.03.
  - **Measurement floor:** the BF16 forward pass changes the anchor log-likelihood by up to 2.5e-4 for random
    directions whose true effect is < 2.4e-6. A one-step effect (2.5e-5 to 1.9e-4) is at or below this floor.
  - **Curvature:** in fp32 the ratios are 0.83, 1.40, 0.70 and 2.89. A fit a*rho + b*rho^2 recovers the predicted
    linear term (a / h.d = 0.62-1.06), but every GRPO displacement has a negative second-order term
    (b = -3.5e-5 to -5.4e-5), about 10^4 times the value for random directions. For the low-alignment update with the
    smallest predicted change, this term is about twice the predicted first-order term after one step.

Interpretation, limited to these 4 displacements on one development checkpoint: first-order alignment with the anchor
gradient does not by itself predict the one-step anchor change of a GRPO update, because the update also lies along
high-curvature directions of the anchor objective. This is descriptive and does not test H1.

| displacement | alignment cos(h, d) | predicted h.d | actual (fp32) | ratio | fitted b |
|---|---:|---:|---:|---:|---:|
| pair 0, high | +0.059 | +1.72e-4 | +1.43e-4 | 0.83 | -3.5e-5 |
| pair 0, low | -0.035 | -1.12e-4 | -1.58e-4 | 1.40 | -4.9e-5 |
| pair 1, high | +0.047 | +1.53e-4 | +1.07e-4 | 0.70 | -4.9e-5 |
| pair 1, low | -0.009 | -2.46e-5 | -7.12e-5 | 2.89 | -5.4e-5 |

Source: `results/bc2rl-e2-taylordiag-dev-s7/taylor_diag.json`, calibration `runs/bc2rl-e2-calib-devlight-s7/calibration.json`.

## 7. E3: online trajectories and paired method comparison

**PENDING** for 11 of 12 branches. Complete: light-BC + GRPO, seed 11 (job 10917856, RTX A6000, 256 rounds,
250 optimizer steps, 6 rounds skipped with no mixed group). Validation pass@1 (144 x 4):

| round | overall | 3 inputs | 5 inputs | 10 inputs |
|---:|---:|---:|---:|---:|
| 0 | 0.042 | 0.120 | 0.005 | 0 |
| 64 | 0.156 | 0.458 | 0.010 | 0 |
| 128 | 0.167 | 0.453 | 0.047 | 0 |
| 240 | 0.198 | 0.526 | 0.068 | 0 |
| 256 | 0.184 | 0.490 | 0.062 | 0 |

The learning signal is sparse. In each round, 32 prompts give 32 groups of 4 attempts. On average only 4.2, 3.7,
3.2 and 2.9 groups were mixed in rounds 1-64, 65-128, 129-192 and 193-256. All-success groups rose from 1.1 to 3.1,
and about 26 groups failed completely throughout. GRPO therefore learned from 9-13% of the sampled groups, and the
signal shrank as 3-input problems saturated (observational, one run).

![E3 trajectory: mixed groups per round and validation pass@1](figures/e3_trajectories.png)

## 8. Retention (H2) and the E4 gate

**PENDING.** Rules (amendment Section 5): attempts 1-7 of the initial 15-attempt evaluation define the initially
consistent subgroup (>= 6/7); attempts 8-15 give an independent baseline; the final checkpoint gets 15 fresh
attempts. Retention claims use k = 1..8 only and need >= 12 problems per stratum.

## 9. Compute

| component | GPU-hours | source |
|---|---:|---|
| consumed before amendment A2 (Stage 0, Stage 1, one branch, base test eval) | 4.08 | sacct + cost ledger |
| E3 remaining, E2, evaluation, reserve | forecast ~36 | DECISIONS 2026-10-25 |

## 10. Deviations, missing runs and limitations

See the amendment, Section 10. Updated at the end of the study.

## 11. Reproduction

```
python -m bc2rl generate  --config configs/data_v1.yaml
python -m bc2rl bc        --config configs/main/bc-broad-s{11,22,33}.yaml
python -m bc2rl online    --config configs/main/on-{light,extended}-{grpo,filtered}-s{seed}.yaml
python -m bc2rl evaluate  --config configs/main/eval-*.yaml | configs/amend1/val15-*.yaml
python -m bc2rl toy       --config configs/mech/toy.yaml
python -m bc2rl mechdata  --config configs/mech/mechdata.yaml
python -m bc2rl mech      --config configs/mech/calib.yaml ; configs/mech/e2-{light,extended}-s{seed}.yaml
python -m bc2rl analyze   --config configs/analysis/final.yaml
```

## References

Numbering follows the revised proposal (`protocol/bc2rl_signal_transfer_retention_proposal.pdf`).
