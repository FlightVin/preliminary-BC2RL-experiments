# Protocol amendment A2 — mechanism study (2026-10-25, 00:30 EDT)

Purpose: record the change of scientific emphasis from the frozen first study (commit `5deefef`,
`protocol/PROJECT.md`, `configs/frozen.json`) to the mechanism study of the revised proposal
`bc2rl_signal_transfer_retention_proposal.pdf` ("When Initial Success Becomes Useful Learning").
This file adds to the frozen protocol. It does not overwrite it. The original protocol stays in
`protocol/PROJECT.md` and `configs/frozen.json`.

Owner instruction: harness `PROJECT.md`, amendment A2 (2026-10-25). Final deadline:
**Tuesday 27 October 2026, 19:00 America/New_York (23:00 UTC)**. It replaces 09:00 EDT.

Question: *When initial policies give comparable success/failure signals, what decides whether an
online update transfers to other problems or damages existing competence?*

## 1. Original frozen protocol and results observed before this amendment

Frozen at 2026-10-24 14:50 EDT (commit `5deefef`), before any test result:

| item | frozen value |
|---|---|
| model | Qwen2.5-0.5B-Instruct @ `7ae5576…`, BF16, LoRA r16/a32/d0 on q,k,v,o,gate,up,down (8,798,208 trainable parameters) |
| data | generator v1 (`configs/data_v1.yaml`); splits dev 144, devdemo 512, bc 2,048, online 4,096, val 144, test 288 (balanced 3/5/10 inputs) |
| BC | AdamW 1e-4, batch 32, stratified order; checkpoints light = 16 steps, extended = 128, continued BC = 256; seeds 11/22/33 |
| online | 32 prompts x G=4 per round, 256 rounds, one optimizer step per round, lr 1e-5 (GRPO and filtered), cap 256, temperature 1, top-p 1, top-k off |
| original matrix | {no-BC, light, extended} x {GRPO, filtered} x {11, 22, 33} = 18 branches; equal-online-round endpoint r256; total-batch endpoints no-BC r256, light r240, extended r128; continued-BC control; 40 test checkpoints |
| test | 288 problems x 15 attempts, eval seed 1; results hidden until the frozen report |

Results observed before this amendment (all on dev or validation data; no test result was read):

- Untouched model, dev, 15 attempts: pass@1 3/5/10 inputs = 0.022 / 0 / 0.
- Learnability gate (dev BC path, seed 7, 128 steps): 0.307 / 0.028 / 0. Generator v1 kept.
- LR calibration (dev): GRPO 1e-5, filtered 1e-5.
- Stage 1 BC validation pass@1 (144 x 4 attempts), light / extended / continued BC: s11 0.031 / 0.118 / 0.139,
  s22 0.038 / 0.097 / 0.158, s33 0.040 / 0.151 / 0.141. 10-input validation pass@1 <= 0.005 everywhere.
- One branch complete: light-BC + GRPO, seed 11 (job 10917856). Validation pass@1 (144 x 4):
  r0 0.042, r64 0.156, r128 0.167, r240 0.198, r256 0.184. By input count at r256: 0.490 / 0.062 / 0.
- Test evaluations complete (hidden): untouched model; light-grpo-s11 endpoints r240 and r256.

Reconciliation of the two "initial" numbers for seed 11 light BC: Stage 1 reports 0.031 and the branch
reports 0.042 at round 0. Both evaluate the same adapter (sha256 `0a58d82f…`) on the same 144
validation problems with 4 attempts. They use different sampling seeds (`seed*7+step` in `bc`,
`tag_seed+100000` in `online`). 3-input successes were 17/192 versus 23/192. The difference is
sampling variation of a 4-attempt panel, not a different checkpoint. Neither number is used for a decision.

## 2. Scope of the revised study

| proposal item | status in this study |
|---|---|
| E0 exact two-task model, estimator and Taylor checks | new; CPU job |
| E1 initial profiles of 6 BC checkpoints + untouched model | from existing Stage 1 and test/val evaluations |
| E2 matched-update intervention (primary) | new; prospective |
| E3 online trajectories: {light, extended} x {GRPO, filtered} x 3 seeds = 12 | 1 complete, 11 to run with the unchanged frozen branch configs |
| E3 retention analysis | new analysis rules (Section 5); prospective for every outcome |
| E4 projection vs norm-matched vs sham | conditional on the gate in Section 6 |
| original no-BC online branches (6) | deferred (proposal Section 1). None ran. No-BC stays a baseline |
| original continued-BC and total-batch endpoints | secondary. Reported only where the original runs produce them anyway (light r240, extended r128, continued-BC test evaluation). Not a headline |

Hypotheses (owner text, A2):
- **H1** Signal availability does not fully determine transfer. At a fixed checkpoint, updates with
  matched observed success/failure composition can have different effects on held-out problems.
  Alignment with an independently defined anchor direction may explain part of the difference.
- **H2** Aggregate improvement can conceal loss of existing competence. Measure retention directly
  on initially consistent problems.
- **H3** Directional constraints may preserve competence beyond shrinking updates. Conditional.

Hypothesis-to-experiment map: H1 -> E2 (primary: paired anchor-check likelihood change; behavioral:
paired probe pass@1). H2 -> E3 retention analysis. H3 -> E4 (only if the gate passes).

Evidence classes in the report: (a) established mathematics (pass@k/pass^k estimators, mixed-group
probability, the GRPO group coefficient, the E0 counterexample); (b) observational evidence (E3
trajectories, retention transitions); (c) intervention evidence (E2, E4). The existence of
intermediate-probability learning signals and of gradient interference is prior work, not a finding.

## 3. Data extension for E2 (same frozen generator)

Three new splits of 48 problems each (16 per input count): `anchorfit`, `anchorcheck`, `probe`. They use
the frozen generator configuration v1 with a new seed (20261025) and global deduplication against
every v1 split by the canonical key. Files: `$PROJ_DATA/data/v1/mech/{anchorfit,anchorcheck,probe}.jsonl`
with `mech_manifest.json` (hashes). This is not a generator revision: the distribution and the rules
are unchanged; only new disjoint draws are made. The v1 manifest is not modified.

Roles (never mixed): anchor-fit traces define the alignment direction h and the selection anchor.
Anchor-check traces measure surrogate transfer. Probe problems measure verified behavior with 15
attempts. Validation (144) supports E3 curves, val15 retention and the E4 gate. Test (288) gives
reported initial/final success and retention. Anchor-check and probe outcomes are never used to
choose pairs, the radius, or checkpoints. The demonstrations are the generator's verified
reference traces; their use is diagnostic extra supervision and is disclosed and costed.

## 4. E2 pre-registration (fixed before any intervention outcome)

Checkpoints: light (16) and extended (128) BC adapters. Seed 11 first; this is a study stage, not an
unreported pilot. Seeds 22 and 33 follow if the seed-11 implementation checks pass and the forecast allows.

1. **Source pool.** The first 1,024 prompts of a fixed permutation (numpy seed 20261025) of the online
   split; the same prompts for every checkpoint. G = 4 completions per prompt with the frozen
   training decoder (temperature 1, top-p 1, top-k off, cap 256), sampling seed 20261025 + checkpoint
   index. Exact binary verification. Mixed group: m in {1, 2, 3} successes.
2. **Group gradient.** GRPO surrogate of the frozen online objective with ratio 1 (same-pass), local
   batch normalization 1 / (32 x 256): g_j = -(1/(32*256)) sum_i A_i grad sum_t log p(y_it), over all
   trainable LoRA parameters (FP32). At most 512 mixed groups are kept (first in pool order).
3. **Anchor direction.** h = gradient of the mean over 48 anchor-fit problems of
   (sum of completion-token log-probabilities of the reference trace incl. `<|im_end|>`) / 256.
4. **Actual displacement.** For a minibatch B of 8 groups (32 completions): g_B = sum_j g_j; clip to
   L2 norm 1 (frozen online setting); one AdamW step from a fresh optimizer state (lr 1e-5,
   betas 0.9/0.999, eps 1e-8, weight decay 0), which is the state of every E3 branch at its fork.
   d_B = -lr * g_hat / (sqrt(g_hat^2) + eps). Alignment a(B) = cos(h, d_B). With a fresh Adam state
   the step is close to lr * sign(-g), so raw-gradient cosine is not the relevant quantity. A unit
   test checks d_B against `torch.optim.AdamW`.
5. **Pairs.** Shuffle the mixed groups within each key (input count, m) with seed 777 and deal them
   round-robin into P = min(8, floor(|M| / 16)) pools. Pools 1-4 form the initial pairs; pools 5-8
   are used only if E2 expands to 8 pairs (decided at the Monday-noon review, from resources only).
   Within a pool, the per-arm count c_k for each key k is the largest-remainder apportionment of 8
   in proportion to the pool counts, with c_k <= floor(n_k / 2). Both arms get exactly c_k groups of
   key k, so the joint (input count, m) histograms match exactly and observed success counts are equal.
6. **High/low selection.** Start: within each key, rank pool groups by cos(h, d_j) (single-group Adam
   displacement); high arm = top c_k, low arm = bottom c_k. Then a deterministic local search
   (at most 3,000 evaluations; swap within key between arms, or with unused pool groups) maximizes
   a(H) - a(L) computed from the actual batch displacements, subject to the length rule: the two arms'
   mean completion lengths differ by <= 5% of the smaller. A pair that cannot meet the length rule is
   reported as unmatched and excluded from the primary analysis. Selection uses only anchor-fit data
   and source-pool statistics.
7. **Norm and radius.** Rescale both displacements to the smaller L2 norm of the pair, then multiply
   both by the common radius multiplier rho. Rho is calibrated on development data before E2 (step 8).
8. **Radius calibration (development only).** Dev light-BC checkpoint (seed 7, 16 steps on devdemo).
   Source pool = dev + devdemo prompts (656). Two pairs built by steps 1-7. Grid rho in
   {1, 2, 4, 8, 16, 32, 64, 128}. Statistic: for each of the 4 displacements, the median over 144 dev
   prompts (one sample each from the unmodified checkpoint) of the per-sequence mean token
   KL(p_before || p_after) over the full vocabulary; take the maximum over the 4 displacements.
   Choose the largest rho whose statistic is <= 0.01. Freeze rho in `configs/mech/frozen_mech.json`
   before any E2 job.
9. **Forks.** Each displacement is applied once to an exact copy of the checkpoint (restored from a
   saved parameter copy and verified by hash before every fork). Forks are never chained.
10. **Endpoints per fork and for the unmodified checkpoint.**
    - selection anchor: anchor-fit mean normalized log-likelihood (implementation and Taylor check);
    - **primary mechanistic endpoint:** change in anchor-check mean normalized log-likelihood;
      paired contrast = Delta(H) - Delta(L);
    - diagnostic: anchor-check log-likelihood split into numeric/operator tokens versus delimiter
      tokens (spaces, `=`, newlines, end-of-turn);
    - **primary behavioral endpoint:** change in probe pass@1 (48 problems x 15 fresh attempts,
      frozen decoder, eval seed 1 + fork tag); pass@k and pass^k for k = 1..15 alongside;
      format-validity and error-type changes reported separately;
    - achieved displacement norm, alignment, sampled-token KL on probe prompts.
11. **Analysis.** Unit = pair within checkpoint. Report each pair's contrast, the mean over pairs per
    checkpoint, and the checkpoint/seed summary. Probe contrasts use a paired whole-problem bootstrap
    (10,000, stratified by input count). A sign-flip test over pairs is a sensitivity analysis only:
    high/low batches are selected, not randomized. Forks share weights and probe problems; they are
    not independent training runs. Opposite-sign pairs (a(H) > 0 > a(L)) are counted; the low arm is
    called "harmful" only for opposite-sign pairs with a negative anchor-check change.
12. **Pre-launch checks (canary on seed-11 light).** Finite gradients; h norm > 0; AdamW equivalence;
    Taylor check on the selection anchor (predicted versus actual change, same sign and ratio in
    [0.5, 2] for rho at calibration); KL statistic <= 0.01; restored-checkpoint hash equality.
13. **Pool replacement (clarification 2026-10-25 01:00, before any E2 outcome).** Pairs use the first
    `n_pairs` pools in pool order whose composition is feasible and whose arms meet the length rule. A pool that
    fails either check is reported and the next pool replaces it. The checks use only source-pool statistics.
14. **Shortage rule.** If fewer than 4 pools can be formed, run the pairs that exist and report the
    count. Do not relax matching or generate more candidates without a recorded forecast update.

## 5. E3 retention analysis rules (fixed before any final 15-attempt evaluation is read)

- Panels: test (288 x 15; initial = `bcpath` evaluation, final = round-256 evaluation) is the reported
  panel. Validation (144 x 15, `val15`) gives the E4 gate and a pre-test view.
- Grouping: attempts 0-6 of the initial evaluation. Initially consistent: >= 6 successes of 7.
  Independent initial baseline: attempts 7-14 (n = 8). Final: all 15 fresh attempts of the endpoint.
- Supported k: retention comparisons k = 1..8 only. Final-only levels at k = 9..15 are descriptive.
  Intermediate 4-attempt validation supports k <= 4 only.
- Minimum subgroup: 12 problems per stratum for a stratum-level retention claim; otherwise
  "underpowered" with counts and descriptive outcomes. The threshold is not weakened after outcomes.
- Transitions: initial class from the first 7 attempts (0, 1-5, 6-7 successes) x final success count.
  Observed consistency is not true p = 1; observed zero is not true p = 0.
- Trajectory associations are observational and do not establish gradient interference as a cause.
- Main E3 contrast: GRPO minus filtered within initialization and seed; then the difference of those
  contrasts between initializations. All seed points shown. Seeds, problems and attempts are distinct
  sources of variation; batches, checkpoints or attempts are never treated as training seeds.

## 6. Metrics and statistics

- pass@k = 1 - C(n-c,k)/C(n,k); pass^k = C(c,k)/C(n,k), C(a,k) = 0 for a < k. Full grid k = 1..15
  for 15-attempt panels; tables at k = 1, 2, 4, 8, 15. Training G = 4 is distinct from evaluation k.
- Average within input count, then equally over 3/5/10. Stratum results are always shown.
- 10,000 paired whole-problem bootstrap resamples, stratified by input count. Curve claims use
  simultaneous bands (max absolute deviation over the k grid within a metric family and comparison).
  Pointwise intervals are labeled pointwise.
- Rare events: when all reliability scores at k are zero, a one-sided bound 1 - alpha^(1/N) with
  alpha = 0.05/(14*3), using one prespecified k-subset (the first k attempts) per problem.
- No favorable k is selected after results; neighboring k values are not independent evidence.

## 7. E4 gate (evaluated on validation only, before the test is opened)

Run E4 only if: on val15, final pass@1 on the initially consistent subset falls by >= 5 percentage
points against its independent baseline in >= 2 extended-BC GRPO seeds, each with >= 12 qualifying
problems; scoring and truncation errors do not explain the fall; and the forecast allows projection,
norm-matched and sham arms (9 branches) to finish before Tuesday 10:00 EDT. Otherwise report
"not run: mechanism not established or resources insufficient".

## 8. Decisions that used results observed before this amendment

| decision | data used |
|---|---|
| LR, round count, cap (original freeze) | dev calibration and dev baseline (Stage 0) |
| keep generator v1 | dev learnability gate |
| scope of E3 (12 branches) and E2 checkpoints (light, extended) | revised proposal text; no outcome |
| execution priority (seed 11 first) | completed run inventory only |
| expectations for E3 (Section 9) | written after seeing the light-grpo-s11 validation curve; disclosed |

No test result has been read. Test results open once, after the analysis code is committed and the
result set is frozen (latest Tuesday 15:00 EDT).

## 9. Expectations recorded now (not tests)

- 10-input strata stay at the floor for every checkpoint and branch; no long-horizon claim follows.
- Learning concentrates at 3 inputs, with a smaller 5-input change; extended-BC starts higher.
- E2 one-step behavioral contrasts may be below the resolution of 48 x 15 probes.

## 10. Deviations from the revised proposal

| proposal | this study | implication |
|---|---|---|
| validation at rounds 0/64/128/256 with 15 attempts at initial/final | frozen branches use 4 attempts at 0/64/128/(240)/256; separate 15-attempt `val15` evaluations of the saved round-0 and round-256 adapters | same information for the endpoints; no change to finished runs |
| up to 8 E2 pairs | 4 pairs per checkpoint initially; expansion to 8 decided Monday noon from resources | lower E2 precision |
| six E2 checkpoints | seed 11 first; seeds 22/33 if checks pass and time allows | possible single-seed E2 |
| per-round anchor audit in E3 | anchor log-likelihood at saved E3 adapters (rounds 0/64/128/256) and an audit-batch alignment at those rounds, if time allows | secondary diagnostic |
| E1 expectations before online outcomes | recorded after one branch's validation curve | disclosed |
| core envelope 240 GPU-h | measured forecast about 40 GPU-h (DECISIONS 2026-10-25) | planning only, not a scientific endpoint |
| anchor/probe splits | new draws from generator v1, seed 20261025 | disclosed data extension |

## 11. Dated clarification D-E2-1 (2026-10-25 11:15 EDT, committed 11:18 in 8183d04; before any E2 outcome on a study checkpoint)

**Event.** Radius calibration (job 10921831, commit 463ecc9, dev light-BC seed 7) chose rho = 1 by the step-8 KL rule
(statistic 0.0048 at rho = 1, 0.021 at rho = 2). The Taylor check at rho = 1 failed: actual / predicted anchor-fit change
= 0.20, 4.45, 0.24, 13.7 for p0-H, p0-L, p1-H, p1-L. The code applies the step-12 Taylor check already at calibration
(stricter than step 12, which applies it to the seed-11 light canary); the seed-11 light job therefore refused to start.
Observed only on development data. No anchor-check, probe, validation or test outcome of any E2 checkpoint exists.

**Diagnosis seen so far.** Actual anchor-fit changes at rho = 1, 2, 4 are not monotone for the H arms (about +4e-5,
-1.8e-4, +1e-5), a size comparable to the predicted first-order change (1.5e-4 to 1.8e-4). This pattern fits a bf16
measurement floor; above rho = 8 a curvature term dominates. An implementation error is not excluded yet.

**Diagnostic (dev only).** `scripts/e2_taylor_diag.py` repeats the measurement on the same checkpoint, anchor-fit set
and 4 saved displacements in bf16 and in an fp32 forward pass, for rho in {0.25, 0.5, 1, 2, 4, 8}, plus 4 random
sign directions of equal norm. rho < 1 is diagnostic only.

**Decision rule, fixed before the diagnostic runs.**
1. If in fp32 all 4 displacements have the same sign and ratio in [0.5, 2] at rho = 1: all anchor log-likelihood
   endpoints, the anchor direction h, the selection and the Taylor check use an fp32 forward pass from now on.
   Sampling, probe evaluation and the KL statistic stay as frozen (BF16 policy). rho stays 1. A fresh calibration run
   in this mode must pass every step-12 check before seed 11 starts.
2. Otherwise E2 is not run as an intervention. The report states "E2 not run: pre-launch Taylor check failed at a
   one-step radius", with the calibration and diagnostic data as a measurement-limited result. The rho grid, the KL
   target and the ratio interval are not changed.
3. If the diagnostic shows an implementation error (for example bf16 and fp32 disagree on h by cos < 0.9, or the random
   directions show a first-order effect), the error is fixed and tested first, and rule 1 or 2 is applied to the
   corrected measurement.

**Outcome (2026-10-25 11:30 EDT; job 10925931, commit 8183d04, dev only).** Rule 2 applies. E2 is not run as an intervention.
- No implementation error: in fp32, h agrees with the bf16 h (cos 0.992), and 4 random equal-norm sign directions follow
  the linear prediction (ratio at rho = 1: 1.03, 0.96, 1.02, 1.01; fitted linear term / prediction 0.92-1.08).
- bf16 floor confirmed: the same random directions give changes of up to 2.5e-4 in bf16 (prediction <= 2.4e-6);
  repeated evaluation at theta0 is bitwise deterministic in both precisions.
- fp32 ratios at rho = 1 for p0-H, p0-L, p1-H, p1-L: 0.83, 1.40, 0.70, **2.89** (rule 1 needs all in [0.5, 2]).
  The fitted linear term agrees with h.d (ratio 1.06, 0.95, 1.03, 0.62), but every GRPO displacement has a negative
  second-order term (-3.5e-5 to -5.4e-5 per rho^2) that is comparable to the first-order term at one step for
  low-alignment arms. Random directions have second-order terms of order 1e-9 to 1e-8.
- Not changed after the outcome: the rho grid, the KL target, the ratio interval, the rule.
