# BC2RL: initial success profiles and online learning

## Mission and authority

Read the research proposal PDF in this harness directory before implementation, especially Sections 2, 4–10 and the final ablation table. Record its filename and SHA-256. This file narrows that proposal to a **first arithmetic study** and replaces its Modal dollar budget with the Babel limits below. Preserve the proposal's definitions and loss formulations unless explicitly amended here. Do not execute its full 36-branch plan.

Install this file as `slurm-agentic-engineering/PROJECT.md` (uppercase: the harness reads that name). Read `agent.md`, `state/{STATE,CHECKLIST}.md`, and `cluster/CARD.md`; maintain the harness's scientific state, findings, decisions, and submission ledger. Follow these project restrictions wherever harness defaults conflict. This specification was checked against upstream commit `6eadc267651aba65153be60746f66f1af4b6e169`; inspect the installed revision before using commands.

**Question:** How does task-specific behavior cloning change the initial distribution of success across problems, and how does that distribution affect gains from group-relative RL versus learning only from successful generated solutions?

**Test three hypotheses:** (H1) initial pass@1 rankings need not predict final rankings; (H2) the online objective's effect depends on initialization and available mixed-reward groups; (H3) charging initialization compute can change pipeline rankings. These are hypotheses, not required findings. The schedule-based initial study can inform H3 but does not by itself establish equal-compute pipeline rankings. BC duration is an intervention; accuracy, coverage, and reliability are measured outcomes, not independently randomized factors.

**Completion:** deliver a reproducible, audited report for the matrix below, including null results, failures, and limits. A defensible feasibility result is acceptable if the measurements cannot support the planned comparisons. Do not expand until this report is complete.

## Non-negotiable resource and ownership rules

- Use **only `preempt` / `preempt_qos` for GPUs**. General allowance is zero, including tests, evaluations, tinkers, and standby. Never use `part=auto`, `ae-topup`, or another pool to bypass a limit. CPU-only jobs use `preempt_cpu_qos`.
- The supplied account snapshot implies `24 - 8 (other ae study) - 8 (wcw study) - 2 (buffer) = 6` available GPUs. However, the ae study is entitled to nine preempt GPUs. **Start with a five-GPU BC2RL ceiling**, preserving that ninth slot and the two-GPU buffer. A sixth requires an explicit release of the other study's reserved slot; idle capacity alone is not permission. Never exceed six without a new owner instruction.
- Before every submission, read current Slurm allocations and requests for shared user `vineethb`, and verify effective GPU/CPU/job/QOS limits. Let `L` be the verified shared preempt GPU limit, `A` and `W` the other studies' committed GPUs, `U` all other non-BC2RL commitments. Enforce `BC2RL_cap = max(0, min(5, L - max(9,A) - max(8,W) - U - 2))`. Change the explicit ceiling/reservation only after the documented release above. Do not infer a larger allowance from the account's permissions.
- Count every nonterminal submitted GPU request: running, pending, held, requeued, configuring, and completing; count requested GPUs for pending jobs and allocated GPUs for active jobs, not job count. Each pending dependency reserves its requested slot. Initially prohibit job arrays; submit only enough individual jobs to fit the cap. The cap includes evaluation, debugging, and canaries. Reserve locally before submission, refresh Slurm immediately afterward, and serialize submissions through one orchestrator. Do not assume another project's controller shares your lock.
- Classify ownership using project path, job ID and ledger, not the `ae-*` prefix alone. Unknown jobs consume external capacity. Never cancel, hold, rename, requeue, or modify other studies' jobs, including `wcw-*` and the other ae study. Jobs `10913178` and `10913179` are historical cancellation requests, not this agent's work: count them until Slurm confirms termination; do not cancel them yourself.
- If shared usage rises, stop submissions and checkpoint/release only enough BC2RL jobs to restore the buffer. If accounting cannot be verified, perform CPU work and report the blocker. Queueing does not make excess requests permissible.
- Use one training process per GPU; no MPS, multi-GPU training, or GPU standby in this phase. Request four CPUs and 32 GiB per GPU job initially; profile before increasing. Limit CPU-only work to one concurrent job, at most four CPUs, within verified shared headroom. Other studies' 32-CPU/192-GiB recipes are not our entitlement.
- Treat the GPU-hour table below as a **proposed planning envelope**, not fixed scientific endpoints or an owner-imposed total grant. During Stage 0, estimate training, evaluation, loading, and preemption overhead on the actual GPU types; think through whether each allocation is adequate and revise it when necessary. Record the forecast, revised envelope, rationale, and completion estimate in DECISIONS before releasing the matrix. Revisit on material throughput or queue changes. The GPU concurrency, pool, ownership, and buffer limits remain hard constraints.
- Count all allocated GPU-hours, including retries, idle allocation time, and preemption rework across job segments. Queueing uses a slot but not GPU-hours. No Modal spending. Reserve forecast allowances before submission and enforce the current documented operational caps across requeues; revisions must be explicit, never silent counter resets. Allocate by measured need, not to exhaust a quota.
- [SUPERSEDED by A2: deadline Tue 27 Oct 2026 19:00 EDT] Deliver usable results by **Tuesday, 27 October 2026, 09:00 EDT (America/New_York)**. Work backward from that deadline and reserve time for evaluation, audit, and reporting. You may revise the proposed envelope within these resource rules to meet that deadline. If the complete matrix will not fit, freeze a smaller common training budget before the main runs, or report partial/feasibility results with missing arms identified. Do not sacrifice the report or extend the study indefinitely.

## Safe harness setup and execution

Use the configured BC2RL branch and remote; resolve existing settings before changing them. Set a unique `AE_REMOTE_ROOT`, such as `'$HOME/ae-bc2rl'`, and `PROJ_DATA=/data/user_data/$USER/BC2RL`. Keep environments, datasets, outputs, and checkpoints under `PROJ_DATA`, never `/home` or `$AE_RUNDIR`. Use experiment IDs `bc2rl-<phase>-<condition>-s<seed>`. Disable GPU standby (`AE_DBG_POOL=0`, `AE_DBG_GEN_POOL=0`); set `AE_DBG_CPU_POOL=1` at most, isolate its name, and count it against the CPU-only limit.

Do not run the default general GPU smoke test. Make a BC2RL preempt-only equivalent using `templates/exp.sh`, verifying GPU visibility, compute storage, pinning, and actual QOS. CPU pinning/dependency tests remain applicable. Implement and test a project submission guard for the rules above; the harness's account-wide quota check is insufficient. Preserve dirty-tree refusal and commit/recipe pinning. Every GPU recipe explicitly declares `gpus=1 part=preempt`, resources, and a wall limit no longer than eight hours or the verified site limit, whichever is smaller.

Use one canary per new run family; inspect a real loss/metrics record before releasing replicas. Dependencies require successful artifacts, because harness dependencies use `afterany`. Never let a failed parent launch a child without validated inputs. Monitor through `ae-watch`; use Slurm as job-state truth. Retry only after diagnosis, under the same resource budget.

Implement atomic, periodic checkpoints at most ten minutes apart and at stage boundaries. Save adapter, optimizer, scheduler, RNG and sampler states, counters, configuration, and cost ledger. Integrate signal handling without replacing the harness's requeue traps. Test interruption and resume: no duplicated accepted updates or evaluation records, no reset compute counters. Charge discarded/repeated work. A done-predicate must validate complete artifacts and matching configuration hashes, not merely directory existence.

## Fixed scientific protocol

**Task.** Use the proposal's arithmetic construction task with **3/5/10 inputs**, requiring exactly **2/4/9 operations**, respectively. Generate positive input integers 2–20 and solvable targets using binary trees and `+`, `-`, `*`; use the proposal's target/intermediate bounds and rejection rules. The verifier consumes a multiset of operands, checks each integer operation, and requires exactly `n-1` operations ending at the target. Accept any valid trace; give reward 1 only for full success, otherwise 0. No partial-credit or format reward. Never execute generated code. This is terminal-reward sequence generation; operation count is distinct from token horizon and is not a robotics claim.

**Data.** Freeze one broad generator distribution and canonical deduplication key `(sorted input multiset, target)`. Use disjoint development (144 problems; 48 per input count, plus 512 disposable demonstrations), BC (2,048), online (4,096), validation (144; 48/horizon), and test (288; 96/horizon) sets. Balance horizons as closely as possible. Share manifests across seeds; hide online reference traces from the learner. Hash manifests, prompts, verifier, and generator. Keep test results inaccessible until the protocol and inclusion rules are frozen.

**Model and initialization.** Pin `Qwen/Qwen2.5-0.5B-Instruct` and tokenizer revisions. Use BF16 and LoRA rank 16, alpha 32, dropout 0 on attention q/k/v/o and MLP gate/up/down projections; freeze base weights. For seeds 11, 22, 33, save a broad BC path at 0.25 and 2 epochs, with stratified first-quarter exposure. Include the untouched instruction model as no-task-BC. Its initial predictions are shared across seeds; do not count three identical initial models as replication. Reset optimizer state when branching online. Use completion-only BC cross-entropy, AdamW at 1e-4, weight decay 0, gradient clipping 1.

**Sampling.** Use 128 fresh completions per rollout round: 32 prompts × 4 independent samples. Temperature 1, top-p 1, top-k disabled, repetition penalty 1, EOS stopping, a provisional 256-token completion cap. Check that valid nine-operation traces fit; raise the common cap to 512 before protocol freeze if truncation warrants it. Reprofile evaluation and use the same frozen cap in all compared arms. Use one common development-only format example in all main arithmetic prompts. The no-BC arm is therefore not zero-shot prompting. Truncation is failure; log lengths and reasons.

**Online objectives.** Implement the proposal's exact Section 7 equations and test them against a hand-computed batch:

- GRPO: `A=(r-group_mean)/(population_group_std+1e-6)`. Sum the clipped token-ratio surrogate over completion tokens and divide by `128*Lmax`; clip radius 0.2, KL 0, no critic. Mask prompt/padding only. Skip the optimizer step if every advantage is zero.
- Filtered supervision: retain successful fresh completions `S`; minimize `-sum(log p(tokens))/(|S|*Lmax)`. Skip the step if `S` is empty. No replay or demonstrations mixed into online updates.
- One full-batch accumulated optimizer step per fresh rollout batch for both methods; no inner epochs. Ratios should initially be 1, so clipping is inactive in this configuration. This tests relative credit versus positive-only imitation, not PPO trust-region benefits. Freeze dropout and synchronize rollout weights; detect stale-policy ratios.
- Calibrate at most learning rates `{5e-6,1e-5}` per objective using the same disposable development initialization and equal rollout-round budgets. Choose development pass@1, smaller rate on a tie. Use AdamW, weight decay 0, gradient clipping 1. Freeze one rate per objective for all initializations; log successful-sample normalization as an intentional scale difference. No outcome-driven sweep.

## Execution order and stop rules

The initial envelope is **224 allocated GPU-hours**; it is a forecast to examine and revise, not an experimental stopping axis. The higher evaluation allowance reflects 15 attempts. At most 46 unique checkpoints × 288 problems × 15 attempts = **198,720 completions** before checkpoint reuse: up to 50.9 million output tokens at a 256-token cap, or 101.7 million at 512. This excludes development and intermediate validation. Forecast from measured end-to-end throughput, including loading and verification, separately for available GPU types. Preserve three paired seeds and final evaluation ahead of optional diagnostics. If the full core cannot fit the deadline, state what can be established from the completed subset; never quietly drop failed or slow seeds.

| Stage | Proposed GPU-hours | Work and exit condition |
|---|---:|---|
| 0. Simple baseline study and implementation | 16 | First audit the untouched model on development problems with 3/5/10 inputs and both zero-example and common-format prompts. Apply the suitability gate below before any large training run. Test verifier, loss, estimators, and restart; profile available hardware. Use at most one disposable light-BC path to check learnability and calibrate learning rates. Freeze tasks, splits, batch sizes, round/epoch limits, and revised resource forecast. |
| 1. Initial policies | 12 | Train three broad BC paths; save 0.25/2-epoch checkpoints. Verify exact forks and validation profiles. |
| 2. Main comparison | 108 | Run `3 initializations × 2 objectives × 3 seeds = 18` branches to the common rollout-round limit below. Interleave methods/seeds. Six hours per branch is only a rough planning average; slower hardware must not receive fewer rounds merely to match hours. |
| 3. Continued-BC control | 12 | Continue three original BC paths to the common total training-batch budget below, using only the original demonstrations. No online data or best-test selection. |
| 4. Locked evaluation and analysis | 52 | Evaluate initials and prespecified round/batch endpoints with 15 attempts; compute paired results, uncertainty, failures, and final report. Reuse evaluations only for identical checkpoint hashes and protocols. |
| Reserve | 24 | Preemption recovery and required evaluation; justify transfers or envelope revisions before use. No additional scientific arms without an explicit protocol amendment. |

**Start with a task that leaves room to learn.** Before BC or RL, measure the untouched instruction model on the 144 development problems with 15 independent attempts per prompt condition. Report all three metrics by input count, format validity, truncation, and sampled errors. The common-format prompt is the operative baseline; zero-example prompting is a diagnostic. Do not mistake inability to follow output syntax for inability to solve arithmetic. Validate representative successes and failures independently of the training verifier.

Use provisional engineering targets, not claims about known model performance: seek at least one 5- or 10-input stratum with roughly **5–60% no-BC pass@1**, non-ceiling pass@4, and enough successes to inspect. Flag >80% no-BC pass@1 as too easy for the main learning claim. Keep an easy 3-input stratum as a labeled control; it must not dominate conclusions. A low pass^4 can still be a measurement floor even when pass@1 is informative. Inspect per-problem success counts rather than approving a task from aggregate accuracy alone.

If all strata are already easy, do not launch the matrix. Make one documented, development-only generator adjustment within the same 3/5/10-input formulation, such as broader input magnitudes or a changed operator mixture, and rerun the baseline. If strata are all near floor (<2%), first check verifier defects, formatting (>20% invalid), and truncation (>5%). Then use the disposable light-BC path to distinguish initial unfamiliarity from an unusable task. Permit one generator revision overall before regenerating and freezing all splits; never select settings for a desired method ranking. Keep a hard 10-input stratum as a reported failure regime if others are informative. If meaningful headroom and learnability cannot be established, deliver the feasibility evidence rather than silently switching models or tasks.

**Training boundaries, independent of GPU speed.** Use a fixed effective BC batch of 32 examples and 128 generated completions per online rollout round. One BC epoch is 64 batches on 2,048 examples; light BC is 16 optimizer steps (0.25 epoch), extended BC is 128 (2 epochs). Accumulation may vary with hardware, but effective batch, example order within a seed, and objective reduction must not. One online round samples 32 prompts × 4 completions and permits at most one optimizer step. Count rounds even when no usable gradient exists; log actual optimizer steps separately so a zero-signal policy cannot sample indefinitely to meet an update count.

Propose **256 online rollout rounds** per branch, with validation/checkpoints at rounds **64, 128, and 256**. This gives 32,768 generated completions per full branch. During Stage 0, assess whether these limits can reveal learning and fit the measured throughput and deadline; revise the common limits if necessary and freeze them before the main matrix. Choose limits using development behavior and feasibility, not final-test rankings. Resume to the same cumulative round target after preemption; resource limits can interrupt a run but must not redefine its scientific endpoint.

For a secondary initialization-allocation comparison, propose **256 total training-batch opportunities**: BC batches plus online rounds. Thus no-BC, light-BC, and extended-BC policies receive respectively **256, 240, and 128 online rounds** at that endpoint. Save these additional checkpoints during the same runs. Continue the pure-BC controls to **256 BC steps (4 epochs total)**, including their original prefixes. If the common total budget is revised, recompute all endpoints from `online rounds = total budget - BC steps`; keep it greater than 128 so every initialization has an online phase, or explicitly revise the comparison before launch.

**Interpret this comparison precisely.** A BC batch and an online round are not equal compute: they differ in examples, generated tokens, and forward/backward work. Call this an equal training-batch-opportunity comparison, not equal FLOPs, GPU-hours, or dollars. The primary comparison fixes online sampling exposure; the secondary fixes this explicit training schedule while including initialization. Report BC steps/epochs, rollout rounds, actual optimizer steps, generated/processed tokens, allocated hours, GPU type, and rework for every pipeline. Charge the full initialization prefix to each standalone pipeline even when reused in the project ledger. Hardware-normalized compute-efficiency claims require an additional justified work/FLOP estimate; otherwise H3 remains descriptive or unresolved. Do not manufacture equal-compute claims by summing unlike GPU-hours. Keep research-only evaluation costs separate from training and account for any selection evaluation explicitly.

## Evaluation and interpretation

Collect **15 independent attempts per test problem** for each unique initial checkpoint and each branch's frozen equal-online-round and total-training-batch endpoints, plus the continued-BC endpoints. Use validation for intermediate curves. Freeze all inclusion choices before opening test results. With `c` successes among `n=15` attempts, estimate:

- `pass@1 = mean(c/n)`;
- `pass@4 = mean(1 - choose(n-c,4)/choose(n,4))`;
- `pass^4 = mean(choose(c,4)/choose(n,4))`, with `choose(a,4)=0` for `a<4`.

Average within each horizon, then weight horizons equally. Report all three metrics together; pass@1 is the primary ranking metric. Pass@4 measures at-least-one success in four tries; pass^4 measures all-four success. Neither finite sampling nor a zero estimate proves capability absence or zero true reliability. Include pass^2 as a diagnostic from the same samples. Do not substitute plug-in powers of empirical accuracy.

Show all three training-seed outcomes and paired method differences. Bootstrap whole problems with all attempts retained, stratified by horizon, using paired indices across conditions (10,000 resamples). Label these intervals as task-sampling uncertainty conditional on trained seeds, not precise population training-seed inference. If all observed pass^4 scores are zero, do not report a degenerate bootstrap interval as certainty: use the proposal's preselected-quartet rare-event bound and report the measurement floor. Do not add attempts selectively to promising methods.

Log mixed/all-success/all-failure group fractions, successes, valid-prefix length, error types, completion lengths, entropy, held-out BC loss, gradient norms, skipped updates, token counts, and compute. For G=4, `mixed_probability = pass@4 - pass^4` under the same prompt distribution and independent sampling. This identity does not prove mixed groups cause improvement. Analyze initial-to-final profiles and horizon interactions; do not infer an independent causal effect of reliability from BC-duration correlations. For discovery/retention diagnostics, preassign the first 7 initial attempts to grouping and the remaining 8 to independent baseline estimation. Compare fresh final attempts; do not reuse grouping outcomes as the baseline. These observed groups do not establish true per-problem probabilities of zero or one.

**Deferred:** restricted demonstrations, G=8, GSM8K, robotics, larger models, KL/replay sweeps, multi-step PPO updates, and adaptive BC stopping. Recommend at most two next experiments from the evidence; do not run them in this phase. The first report tests the initialization/objective interaction; it cannot establish the proposal's demonstration-structure causal claim or cross-domain generality.

## Reproducibility and final deliverables

Implement a small CLI in the BC2RL project: `python -m bc2rl {generate,bc,online,evaluate,report} --config <yaml>`, with explicit resume/run identifiers. Document actual commands, environment lock, model revisions, seeds, manifests, and verifier version. Use harness experiment templates and commit-pinned jobs. Keep a versioned copy of this protocol and final frozen configurations in BC2RL; the harness workspace itself is ignored by default.

Each run writes full attempt-level records and checkpoints to `PROJ_DATA`, and a ≤2 MiB distillate to `$AE_RESULTS`: `summary.json`, compact metrics CSV, provenance/config hashes, completion status, and paths/hashes for large artifacts. Include prompt ID, checkpoint, seed, attempt ID, decoding settings, reward, failure reason, token counts, and cost in full records. Harvest summaries; never silently truncate evidence to satisfy the cap.

Deliver `reports/initial_study.md` and `reports/initial_study.pdf`, plus machine-readable result tables and figure-generation commands. Include the frozen run matrix and deviations; measured suitability; all metrics by initialization/method/seed/horizon; equal-online-round and total-training-batch comparisons; coverage–reliability and compute curves; actual GPU-hour ledger; failure/rare-event analysis; supported, contradicted, and unresolved claims; and at most two concrete follow-ups. Cite the proposal's relevant primary sources and distinguish inherited claims from new measurements.

Before completion, audit the report against raw artifacts and Slurm accounting, independently recompute a sample of metric and cost rows, verify every scheduled arm is complete/failed/not-run with a reason, and review prose and figures. Record the audit, update FINDINGS/STATE/CHECKLIST, and release only BC2RL-owned allocations. A usable final report is mandatory even if feasibility or resource gates stop training early.

## Owner amendments

### A1 — 2026-10-24 (supersedes the `reports/initial_study.pdf` deliverable)
> "Oh write the report as an MD file and not PDF - include plots into it for better understanding. Update state and checklist and proceed autonomously."

Effect: deliver `reports/initial_study.md` with the figures embedded as images (committed PNG files
under `reports/figures/`), plus the machine-readable tables. No PDF is produced.

### A2 — 2026-10-25 00:00 EDT: scientific pivot to the mechanism study; new deadline (supersedes the 09:00 EDT deadline and its cutoffs, and the PROJECT "Question"/"Test three hypotheses" emphasis)
Source: owner message of 2026-10-25 (full text in the session; faithful summary below). Protocol detail:
`protocol/AMENDMENT-A2-2026-10-25.md` in the project repo (commit 90db8b9). Proposal:
`bc2rl_signal_transfer_retention_proposal.pdf` sha256 2603f2337d82bef4d183446e2d10a00805064d69db8118ac28780cdb3ca2ad11.

- **Deadline:** "Tuesday, October 27, 2026, at 7:00 PM America/New_York (23:00 UTC)". Planning checkpoints:
  Sun noon (resources, protocol, minimum viable plan); Mon noon (feasibility of all seeds and optional work; freeze scope);
  Mon evening (complete report draft, missing results marked); Tue 10:00 (no new exploratory branches; only planned
  completions, essential evaluation, correctness fixes); Tue 15:00 (freeze result set); Tue 15:00-19:00 (analysis,
  figures, reproducibility, accounting, editing). A passed checkpoint is reviewed immediately.
- **Question:** "When initial policies provide comparable success/failure signals, what determines whether an online update
  transfers to other problems or damages existing competence?" Hypotheses H1 (signal availability does not fully determine
  transfer; matched-composition updates; anchor alignment), H2 (aggregate improvement can conceal loss of competence;
  measure directly), H3 (directional constraints beyond shrinking; conditional). Distinguish established mathematics,
  observational evidence and intervention evidence; do not present mixed-signal or interference existence as discovery.
- **Preserve** the existing study, valid runs and the original protocol history; preserve test blinding; never use hidden
  test results for checkpoints, conditions or reporting priorities. Reconcile facts from artifacts.
- **Babel (unchanged, restated):** preempt GPUs only; at most five BC2RL GPU slots, subject to any tighter restriction;
  no general GPUs, no `ae-topup`; count running+pending allocations, array concurrency, evaluations, canaries, standbys;
  a pending job occupies a slot; never infer more allowance from idle capacity. Never touch `wcw-*`, other `ae-*`, or
  jobs not established as BC2RL. CPU-only jobs on the authorized CPU QoS and its limits. BC2RL-owned jobs may be submitted,
  resumed, checkpointed, moved within permitted resources or cancelled; preserve checkpoints/audit records first; do not
  cancel a nearly complete run to change priorities. Bounded worker allocations that run several registered experiments
  sequentially are permitted; each keeps its own config, seed, checkpoint, logs, accounting; no idle standby, no duplicate
  execution, no unsafe concurrent writes. Inspect and reconcile the queue before scheduling changes.
- **Budget:** reconcile the ~30 GPU-h envelope; forecast separately consumed, remaining existing comparisons, new
  mechanism experiments, evaluation/analysis, preemption reserve, from measured throughput per GPU type. GPU-hours are
  accounting, not an endpoint or entitlement. Preserve frozen steps/rounds/epochs/exposure/decoding for comparable runs;
  no opportunistic shortening presented as full endpoints; report common-intermediate comparisons separately. Reduce
  optional scope before the minimum coherent set.
- **Order:** A audit suitability/checkpoints per input count (success, format/parse failures, invalid arithmetic or input
  reuse, truncation, success-count distributions, mixed-group frequency); no generator change, no easier substitute; no
  long-horizon claim if 10 inputs stays at floor. B complete one seed's paired init x method comparison and BC controls
  under the original matrix, then other seeds. C E2 matched-update experiment (light and extended BC, seed 11 first, then
  22/33): frozen decoder and G; exact verification; pairs matched on joint (input count, m in {1,2,3}); mean length within
  tolerance with residuals; separate anchor-fit set defines the direction; alignment from the actual optimizer displacement
  incl. Adam; equal norms in a pair; development-calibrated small-update rule; fresh copy per update, identical optimizer
  handling, no chaining; evaluate on independent anchor-check and behavior probe; roles separate; never select on
  anchor-check/probe; **four disjoint pairs per checkpoint initially, eight if resources permit** (recorded before
  outcomes); keep batch size and bounded pool unless documented; report shortages; primary local evidence = paired
  anchor-check log-likelihood change; verified behavior is separate; check stability, norms, KL, directions; CPU toy model
  as a sanity check. D retention on existing trajectories: first 7 attempts define consistent subgroup, remaining 8 the
  independent baseline, fresh endpoint attempts final; proposal threshold and minimum size; underpowered labels;
  observational. E optional projection experiment only after core work, with gate met and all controls (ordinary GRPO,
  projection, norm-matched, sham); otherwise defer.
- **Metrics:** pass@k and pass^k throughout, not organized around k=4; full panels k=1..15 with values at 1, 2, 4, 8, 15;
  combinatorial estimators; G distinct from k; retention k<=8; intermediate 4-attempt k<=4; report high-k floors; no
  post-hoc k; paired problem-level analysis by input count; separate problem and seed uncertainty; simultaneous intervals
  and rare-event guidance.
- **Operations:** keep STATE/CHECKLIST as the single sources of truth (completed, running, queued, blocked, deferred,
  cancelled; checkpoint/config IDs; dependencies; acceptance checks; resource use; amendments; next actions; what each claim
  still needs). One experiment registry linking every result to split, config, seed, checkpoint, code revision, job ID
  (`state/REGISTRY.md`). Progress/timeline update every 4 hours.
- **Report:** update `reports/initial_study.md` (Markdown with plots): what was tested/untested; task-difficulty adequacy;
  paired method/init results; matched-update results and limits; retention across k and input counts; optional
  intervention if completed; alternative explanations and negative findings; exact compute, missing runs, deviations;
  reproduction commands and paths. "A complete negative or measurement-limited result is acceptable. Unsupported claims of
  mechanism, novelty, generality, or long-horizon capability are not." Do not expand resource permissions or silently
  change the protocol; prefer a small, interpretable, reproducible study.

### A3 — 2026-10-25 10:10 EDT: laptop work while Babel jobs pend (faithful record of the owner message)
- Authorized: run lightweight BC2RL work on this laptop while Babel jobs remain pending: CPU checks, data preparation,
  analysis, report writing, and small experiments that fit its resources. This overrides `agent.md` "do not run project
  code on the laptop" for BC2RL only, within these limits.
- Keep **all project-created files, environments, caches, datasets, checkpoints and logs within one dedicated directory
  totaling no more than 20 GB** (chosen: `~/bc2rl-local`). Monitor usage; do not disrupt normal laptop use.
- Keep watching Babel. **As soon as an authorized preempt GPU allocation starts:** checkpoint and stop local experiments,
  transfer all new project data and artifacts to Babel, verify integrity, resume computation there; avoid duplicate execution.
- After verification, delete the laptop temporary project directory and any project-created caches; do not delete
  pre-existing files or shared environments. "This migration and cleanup are explicitly authorized."
- Preserve the Babel allocation limits, the protocol and the Tuesday 19:00 EDT deadline. Record local work, resource use,
  migration, verification and cleanup in STATE, CHECKLIST and the next progress report.

### A4 — 2026-10-25 22:55 EDT: result freeze now (owner choice "Option 2 - freeze now"; supersedes the Tue 15:00 freeze time and the Mon 12:00 scope-freeze review window)
Context: no BC2RL job was running and no GPU work remained in scope. The orchestrator offered three options: freeze at Mon
12:00, freeze now, or keep Tue. Owner message (verbatim): "Option 2 - freeze now." Effect: the result set is frozen at
project commit 22e6816; the final analysis (allow_test) runs once now; delivery moves to Mon 2026-10-26 (deadline Tue 19:00 unchanged).
No new branches, analyses or checkpoint choices after the test read.

### A5 — 2026-10-26 (approx. 03:30 UTC): separately pre-registered extension (supersedes the A2 deadline, the ~30-40 GPU-h envelope and A4's "no new analyses"; the frozen study stays frozen)
Owner message (faithful record; the owner first sent a version with a ~40 GPU-h envelope, then replaced it before any action with this one):
- "Continue BC2RL with a **separately preregistered extension** testing whether extended BC makes intermittently
  successful problems more vulnerable to online training than light BC. Read the revised proposal PDF and current project
  state first. Preserve the frozen study and its exploratory/post hoc labels. **The Tuesday deadline and cumulative
  GPU-hour ceiling are removed; allocation limits remain binding.**"
- 1 Audit the existing results: inconsistent contrast counts, duplicate pass@1/pass^1 checks, retention statements,
  uncertainty-band scope; correct conclusions that exceed the evidence.
- 2 Independently confirm the emerging finding: existing six BC checkpoints and 12 online endpoints on fresh, disjoint
  3-input problems plus a smaller 5-input panel. Before viewing new outcomes, preregister sample sizes, primary
  comparisons, estimators, exclusions and stopping rules; choose sample sizes for useful precision from measured throughput.
- 3 Improve measurement: more initial attempts than 15, separating subgroup selection from baseline estimation; coverage
  loss, consistency gains, loss conditional on initial success, sensitivity to noisy baseline estimates; pass@k/pass^k
  across supported k; paired uncertainty preserving shared problem identities; seed-specific results; distinguish
  fresh-problem confirmation from independent training replication.
- 4 Repair and execute E2: FP32 measurement; development-only calibration across smaller update radii; absolute Taylor
  error, curvature, predicted vs observed changes. If the assay passes prespecified validity checks, run matched,
  equal-norm update comparisons across light/extended checkpoints and seeds with independent selection and evaluation
  anchors. Document failure if it stays unreliable; "do not relax checks to obtain a positive result."
- 5 Expand according to unresolved questions (training seeds, targeted controls) with preregistration before each
  confirmatory stage; follow the PDF's questions without assuming its hypotheses hold; stop when further experiments are
  unlikely to change the interpretation materially.
- Compute: "use preempt only, with **at most five BC2RL GPU slots**, counting running and pending allocations and array
  concurrency. No general GPUs, `ae-topup`, or interference with other projects. There is **no cumulative GPU-hour cap**,
  but maintain accounting and update the resource forecast before each stage. Use fixed rounds/steps and controlled
  sampling budgets for scientific comparisons, not equal GPU-hours."
- Records: update PROJECT.md, amendments, registry, state, checklist. Monitor through scheduler events and batched checks.
  After implementation and submission: timeline and resource forecast. Update **`reports/extension_study.md` every eight
  hours** (results, uncertainty, blockers, next experiments, resource use, revised completion estimates).
- Before each report update: two code-audit passes (implementation correctness; statistical/protocol validity) and two
  writing-review passes (evidential accuracy; clarity and precision). Every empirical claim traces to an artifact;
  separate observations, interpretations and hypotheses.
Note: no PDF newer than the A2 proposal exists in the repo or workspace; "revised proposal" = `bc2rl_signal_transfer_retention_proposal.pdf`
(sha256 2603f233...ad11, the A2 revision). Protocol detail: project `protocol/AMENDMENT-A5-2026-10-26.md`.

### A6 — 2026-10-26 (approx. 23:00 UTC): mechanism follow-up study (new amendment; A5 and its stop rule stay complete and valid for A5)
Owner message (faithful record; the full text is in the session):
- "Continue BC2RL after compaction ... Preserve the completed A5 study and its stop rule; **this instruction authorizes
  a new amendment and follow-up study**." Established finding: online training from extended-BC checkpoints produces
  greater loss of intermittently accessible solutions than training from light-BC checkpoints. "**The next goal is to
  distinguish explanations, not merely reproduce the effect again.**"
- Order: (1) audit the interpretation: change from baseline vs worse final performance; subgroup selection, common
  weighting, paired uncertainty, seed dependence; matching observed counts is not matching true success probabilities;
  correct overstatements without rewriting the historical protocol. (2) Resolve truncation and characterize failures:
  re-evaluate existing checkpoints on a separate diagnostic panel at the original and a larger token cap, other decoding
  fixed; truncation = failure under each cap; decompose losses into truncation, formatting, illegal operands, incorrect
  arithmetic, valid chains reaching the wrong target; which types account for the extended-light difference.
  (3) Harder problems and easier-problem losses: same starting checkpoints; original mixed training vs 3-input-only
  training; a control that holds 3-input exposure fixed while omitting harder-problem updates, distinct from an
  equal-total-rollout comparison; GRPO first, filtered supervision if needed. (4) BC exposure and initial competence:
  32- and 64-step checkpoints on the same BC paths; fixed training budgets; independent development measurements define
  comparisons with overlapping initial competence; fresh problems for confirmation; match more than aggregate pass@1;
  report residual mismatch; do not force a matched comparison when overlap is inadequate. (5) Redesign E2 only if it can
  resolve the remaining mechanism question; new development data; absolute-plus-relative error criterion; validate
  precision, curvature, actual optimizer displacement, independent held-out effects; freeze the gate before
  confirmation; failed validity is not evidence against interference; reference-trace likelihood alone is not evidence
  of preserved verified reasoning.
- Questions: intermittent solutions less likely vs hidden by length/format; same-difficulty vs cross-difficulty;
  BC exposure beyond the initial success distribution (gradual / threshold / non-monotonic); problem structures or
  strategies; filtered vs GRPO (test a direct method interaction before claiming a difference); can controlled update
  effects predict later behavioural losses. Use them to choose discriminating experiments, not an unrestricted sweep.
  Before each stage record competing explanations, predicted outcomes, primary comparison, sample size, uncertainty,
  stopping rule. Separate exploratory diagnosis from confirmation on fresh data. Reuse valid checkpoints; add training
  seeds where they materially improve inference.
- Resources: "preempt only; **at most five BC2RL GPU slots**, including running and pending allocations and array
  concurrency, subject to any tighter current authorization. No general GPUs, `ae-topup`, or interference with other
  projects. There is no cumulative GPU-hour cap or Tuesday deadline." Accounting and stage-level forecasts; controlled
  steps and sampling budgets.
- Records: update the amendment, PROJECT.md, registry, state, checklist; implement and launch autonomously; after
  submission report the matrix, rationale, timeline, forecast. Update **`reports/mechanism_followup.md` every eight
  hours**. Monitor through scheduler events and batched checks, low tokens.
- Before each substantive report update: two code-review passes (implementation; statistical/protocol correctness) and
  two writing passes (evidential integrity; clarity); reuse audits of unchanged components; conclusions traceable to
  artifacts; report negative findings; stop expanding when the remaining uncertainty is characterized or further
  experiments would not distinguish the leading explanations.
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` (stages Y1-Y5).

### A6-G — 2026-10-27 ~15:00 EDT: bounded GSPO experiment inside A6 (owner; faithful record; adds to A6, supersedes nothing)
- Add a bounded GSPO experiment to the current follow-up amendment without displacing the existing mechanism work.
  Read the GSPO paper (arXiv 2507.18071), inspect the implemented GRPO objective, and pre-register the question:
  "Does sequence-level weighting and clipping change coverage loss or its dependence on BC initialization, beyond
  effects from response-length normalization and rollout reuse?"
- 1. Validate the implementation first: GSPO's length-normalized sequence ratio with frozen behaviour-policy log
  probabilities; test masks, EOS handling, clipping, numerical precision and gradients; verify that GSPO and
  response-length-normalized GRPO agree at the first on-policy update; preserve the constant-normalization GRPO as a
  separate reference.
- 2. Informative comparison: light and extended BC for one prespecified seed; token-level GRPO vs GSPO with the same
  response-length normalization, one vs four full-batch optimizer steps per rollout batch; behaviour probabilities
  fixed through batch reuse; prompts, group size, rollout rounds, decoding, reward and optimizer settings constant where
  appropriate; calibrate clipping ranges separately on development data (identical thresholds need not impose
  comparable constraints).
- 3. Measure pass@k / pass^k, intermittent-success coverage loss, output length, truncation, KL, ratio distributions,
  clipping activity and update norms; compare equal-rollout endpoints; report optimizer steps and actual compute
  separately; if ratios stay near one and clipping is negligible, state explicitly that the experiment barely
  exercises the distinction.
- 4. Expand only after validation: the first seed is development; if valid and informative, freeze the configuration
  and replicate over at least three prespecified training seeds with independent confirmatory evaluation; test the
  method x BC-initialization interaction, not only overall accuracy; do not attribute a normalization effect to
  sequence-level clipping.
- Resources: preempt only; at most five BC2RL GPU slots across all studies (pending allocations and array concurrency
  count); no general GPUs, `ae-topup` or other-project interference; stage-level forecasts; no cumulative GPU-h ceiling.
- Update the amendment, registry, state and checklist; include the experiment in the 8-hourly report cycle with the
  established code and writing reviews; proceed autonomously; report negative or uninformative results explicitly.
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` Section 12 (pre-registered at 9abd2a8).

### A6-G-CTPO — 2026-10-27 ~15:20 EDT: add CTPO to the stage-G comparison (owner; faithful record; adds to A6-G)
- Extend the registered GRPO-GSPO comparison to CTPO (paper arXiv 2605.07331; official code horizon-llm/CTPO); read the
  current amendment and experiment state first; integrate without duplicating runs or displacing the mechanism study.
- Question: "Does cumulative-prefix correction change coverage loss and its dependence on BC initialization, compared
  with token-level GRPO and sequence-level GSPO?"
- 1. Verify the formulation: theorem assumptions, loss definition, implementation; distinguish guarantees for ideal
  importance-weighted estimators from practical training with group-estimated advantages and clipping; do not claim
  our estimator is unbiased without establishing the assumptions.
- 2. Implement faithfully: position-adaptive clipping; cumulative ratios through log-space sums; verify gradient flow
  (incl. detach operations) against the reference code (a plain derivative of a cumulative product adds prefix-gradient
  terms); test masking, EOS, numerical stability and on-policy behaviour on a small exact example.
- 3. Reuse the controlled comparison: same light/extended checkpoints, prespecified dev seed, 1 vs 4 optimizer steps
  per rollout batch; frozen behaviour probabilities; control reward, advantages, normalization, sampling and training
  exposure; document necessary method-specific differences; calibrate clipping on development data.
- 4. Measure pass@k / pass^k, intermittent-success coverage loss, truncation, output length, KL, update norms;
  cumulative log-ratio distributions and clipping rates by token position; if an apparent benefit emerges, a targeted
  fixed vs position-adaptive clipping ablation.
- 5. Confirm selectively: after validation and development, freeze and replicate informative comparisons over >= 3
  prespecified seeds; method x BC-initialization interaction with paired uncertainty; report null results and cases
  where negligible policy mismatch leaves importance-weighting differences largely untested.
- Resources as A6-G (preempt only; 5 BC2RL GPU slots across all experiments incl. pending and array concurrency; no
  general GPUs, `ae-topup`, other-project interference; no GPU-h ceiling; forecasts and accounting). Update amendment,
  registry, state, checklist; 8-hourly report cycle with the established audits.
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` Section 12.8 (pre-registered at b951f90).

### A6-S — 2026-10-28 ~17:15 EDT: program S — learning signal, transfer, concentration (owner; faithful record; adds to A6; supersedes nothing)
- "Add the following experimental program to `PROJECT.md`, the active amendment, state, and checklist. **Continue
  existing work without interruption.** Reuse valid experiments and checkpoints, but create new data or training
  conditions where needed." Goal: "distinguish insufficient learning signal, harmful transfer between problem classes,
  and concentration onto fewer solution strategies. Then test an intervention that addresses the supported
  explanation." No positive result required. Each stage specifies primary comparison, controls, evaluation
  population, uncertainty procedure and stopping rule before confirmation.
- 1. Resolve the cross-difficulty question first (M / MASK / T3): per-BC-init effect of harder-problem updates; whether
  it differs light vs extended; equal-total-sampling vs equal-3-input-exposure. A near-zero init interaction does not
  imply no effect. Paired seeds only under a prespecified precision plan that resolves a meaningful uncertainty.
- 2. Dataset separating structure from difficulty: 3 inputs, existing rules and binary verifier; enumerate all valid
  solutions; classes "no multiplication required" / "multiplication required"; record minimum multiplications, number
  of distinct solution forms, necessity of a negative intermediate, input/target magnitude, independently measured
  initial success per starting checkpoint; canonicalize equivalent solutions; count execution traces and mathematical
  forms separately. Do not assume the classes differ only in operator use; inspect overlap on dev data; balanced pools
  with common support where feasible; report unmatched properties. Two evaluation sets (representative fixed panel;
  diagnostic panel balanced over classes and baseline-success ranges), reported separately; fresh confirmation
  problems; report how deduplication/filtering changes the distribution. Optional separate 4-input extension (classes
  from all valid solutions; learnability on dev first); the 3-input study stays distinct.
- 3. Transfer: from identical light/extended checkpoints, A-only, B-only, balanced A/B; established GRPO; fixed rounds,
  prompts/round, group size, decoding, reward, loss reduction; control pool sizes, record repetition. Every arm on
  both classes: composition x evaluation-class table. Paired masked controls (mixed schedule and target-class loss
  unchanged; other class's gradients included vs masked without reallocating budget); specify optimizer state and
  skipped steps; interpret as total effect. Report successful groups, mixed-reward groups, gradient contribution by
  class; equal sampled prompts are not equal signal.
- 4. Diversity: narrow vs broad structural pool with fixed unique problems, completions/rounds, initial success and
  other difficulty variables; separately 64 vs 256 unique problems at fixed completions. Evaluate fresh represented
  structures, withheld structures, representative panel. If diversity and difficulty cannot be separated, report the
  joint intervention.
- 5. Preservation: vulnerable class chosen on dev evidence and frozen; rehearsal set of independent intermittently
  solved training problems with verified starting-policy solutions; online / + targeted rehearsal / + equal broad
  rehearsal (same tokens and coefficient; coefficient calibrated on dev and frozen; fixed online exposure; report extra
  cost). Measure benefits and costs; accuracy-coverage tradeoff plot; earlier-checkpoint comparison. Positive rehearsal
  does not prove gradient interference.
- 6. Strategies (diagnostic): keep generated solutions on a prespecified subset; canonical forms, concentration,
  disappearance, losses on forms with little successful training; equal attempts; account for success rates.
- 7. Optional small frozen-policy "oracle-assisted search" diagnostic, only if a question stays open.
- 8. One dev configuration per family, then frozen confirmation with >= 3 paired seeds; seeds added under a precision
  plan only. Report pass@1 and full pass@k / pass^k, final and change, whole-panel and subgroups, independent
  selection/baseline estimates, failure types, truncation, signal counts, optimizer steps, resources, problem and seed
  uncertainty separately. Priority: finish cross-difficulty; structural transfer; then the diversity or rehearsal
  comparison that best separates the remaining explanations; no full Cartesian product; coordinate with stage G.
- Continue autonomously; a failed gate leads to diagnosis, not relaxed criteria. Preserve preempt-only access, the
  5-slot total BC2RL cap, the stricter shared-account guard, CPU limits and all ownership restrictions (pending
  allocations count). No deadline or GPU-hour ceiling; stage-level forecasts and accounting. Keep PROJECT, amendment,
  registry, state, checklist current; 8-hourly report cycle with the established code and writing reviews.
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` Section 13.

### A6-R — 2026-10-28 ~18:00 EDT: stage R — retention, policy drift, response source (owner; faithful record; adds to A6; supersedes nothing)
- "Continue BC2RL without interrupting active experiments or changing their registered protocols ... Add the following
  comparisons ... through a dated amendment. Reuse completed work where appropriate." Question: "does retention depend
  on how far the policy changes, and on where its successful training responses come from, beyond differences in
  initial competence and subsequent improvement?"
- 1. Drift on existing checkpoints first: reference = each branch's own BC checkpoint; rounds 0, 64, 128, 256 where
  saved (never re-create missing checkpoints as originals); 192 three-input + 64 five-input problems from the opened
  A6 dev panels by fixed hash order; reuse their selection/baseline/outcome attempts; 16 extra reference responses
  per problem. Forward sequence KL(initial || current) over all reference responses incl. failures; sum token
  log-prob differences incl. sampled EOS; explicit handling of capped prefixes; consistent FP32 scoring; verify
  sampling vs scoring probabilities; per-token average is a separate diagnostic; reverse KL needs current-policy
  samples. Relate drift to coverage loss, competence, structure, length, improvement; within-seed trajectories and
  leave-one-seed-out prediction; exploratory; confirm selected relationships on the new runs; no mechanism claim from
  correlation.
- 2. Matched progress: beside fixed rounds, a common pass@1 improvement level chosen on dev data only; within each
  init, the earliest saved checkpoint within +-0.01 absolute; freeze target and rule before confirmation; no common
  level -> unavailable; no extrapolation; no selection on confirmation outcomes; cross-init: matched improvement vs
  matched final accuracy kept distinct; report uncertainty in achieved progress.
- 3. Response source, one supervised loss: S solver cache (one fixed valid trace per problem), I initial-policy cache
  (one fixed verified starting-BC response), F fresh frozen-policy responses, O fresh online-policy responses; S vs I,
  I vs F, F vs O; not GRPO vs supervision. Dev seed 7, then BC 16/128 x seeds 11, 22, 33 = 24 confirmatory branches;
  3-input problems first. Pool per Section 11 of the updated proposal: 256 problems initially solvable by all six
  starting checkpoints, at most 2,048 screened candidates, dev-fixed fallback 128; independent attempts for I caches;
  record exclusions and screening costs; studies initially available solutions only. Fresh sources: until first
  verified success or 64 attempts; one target per occurrence; if F or O fails, mask in all four arms; identical prompt
  order, accepted opportunities, loss denominator, optimizer conventions; >5% masking in a block = failure of the
  exposure-matching criterion (no threshold change after outcomes). 256 rounds, 32 occurrences/round, lr 1e-5, the
  common supervised loss of Section 11; fixed adapter, decoding, optimizer; save 0, 32, 64, 128, 192, 256; each
  four-arm block sequential in one GPU worker with separate optimizer states; count all candidates and cache costs.
- 4. Fresh panels 768 three-input + 128 five-input; 48 initial attempts (24 selection / 24 baseline), 32 per final.
  Primary: F vs O standardized intermittent-success coverage loss at k = 16; also other source contrasts, init
  interaction, whole panel, full pass@k / pass^k; Holm over the three primary source comparisons; problem and seed
  uncertainty separate. Report initial/final/change, drift, failure types, target diversity, tokens, optimizer steps;
  keep fixed-exposure results when matched-progress results differ.
- 5. Complements cross-difficulty, structural-transfer and GRPO-GSPO-CTPO work; no duplication or restart; checkpoint
  analysis and development checks before the new training matrix. Hard limits: preempt only, <= 5 BC2RL slots incl.
  pending, stricter shared-account guard, CPU limits, one submission orchestrator; no general GPUs, `ae-topup`, MPS,
  multi-GPU, other-project interference; no deadline or GPU-h ceiling; measured accounting and a forecast before each
  stage. After implementation and submission report matrix, dependencies, timeline, forecast; 8-hourly report updates
  with the established checks; close uninformative comparisons explicitly; no stopping-rule changes for significance.
- Note (2026-10-28 18:04): the proposal copies in `protocol/` (2026-10-24) have no Section 11 with the common pool or the
  supervised loss; the loss is an open item that blocks only the R3 training matrix (protocol 14).
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` Section 14 (49fc6fd, 605ccf5).

### Owner amendment A6-R2 (2026-10-29 ~01:45 EDT) — wave-2 decision kept; R3 supervised loss
Faithful record of the owner message:
- Keep the wave-2 decision: do not add Y3 seeds, run filtered Y3, or redesign Y5. Report the harder-task contribution as
  unresolved within the stated bounds, not absent. Keep the new contrasts labeled exploratory.
- This supplies the shared supervised loss that unblocks stage R. For all four response-source arms (solver cache,
  initial-policy cache, fresh frozen-policy responses, fresh evolving-policy responses):
  L(theta) = -(1/(32 x 256)) sum_{i=1..32} m_i sum_{t=1..T_i} log pi_theta(y_it | x_i, y_i<t), where y_i is one verified
  successful response from that arm's designated source; m_i is the shared acceptance mask (if either fresh-source arm
  cannot produce a verified response within its 64-attempt cap, m_i = 0 for that prompt occurrence in all four arms);
  T_i counts actual response tokens incl. a sampled EOS, excluding prompt and padding tokens; a truncated response cannot
  become a successful target. Keep the denominator fixed at 32 x 256, also when occurrences are masked; do not divide by
  accepted-response count, actual token count or individual response length.
- Teacher forcing. Collected responses and acceptance decisions are fixed data; gradients pass only through the trained
  policy's log-probabilities. No reward baseline, advantage normalization, importance ratio, clipping surrogate or KL
  penalty.
- AdamW, lr 1e-5, weight decay 0, gradient-norm clipping at 1. Reset optimizer state when branching from BC. Hold all
  other optimizer settings, numerical precision and trainable parameters identical across arms. Accumulate the complete
  batch loss before one optimizer step. If all 32 occurrences are masked, skip the optimizer and LR-scheduler steps but
  count the sampling round.
- Collect the frozen-policy and evolving-policy targets before updating any arm for that round; apply their common mask
  to all four arms; separate optimizer state per arm. 256 sampling rounds; checkpoints at rounds 0, 32, 64, 128, 192, 256.
- This normalization intentionally differs from the original success-filtered baseline. Record it as the controlled
  response-source experiment, not a reproduction of that baseline.
- Add this definition to the amendment and implementation tests. Verify loss scaling, token masking, shared acceptance
  masks and empty-batch behaviour on a hand-computed batch. Then run the development feasibility checks and proceed
  under the registered gates.
- Continue other authorized work without interruption. Preserve the five-slot preempt-only cap and the shared-account
  guard. Update state and checklist now; include the revised stage-R forecast and the next decision point in the next
  eight-hour report.
Protocol detail: project `protocol/AMENDMENT-A6-2026-10-26.md` 14.9.1-14.9.2 and 13.11.1 (2026-10-29).

### Owner amendment A7 (2026-10-29 16:54 EDT) — stop at 10:00 on 2026-10-30; final report; reset for a fresh run
Faithful record of the owner message:
- At 10:00 on 2026-10-30, stop all tasks. Curate the report with what we know at this stage and what remains to be seen.
- Store PROJECT.md as a report. Revert STATE and CHECKLIST to empty. Keep PROJECT.md as a skeleton and keep the compute
  budget in it. A fresh run starts after that point.
- Deliver these by 11:00 on 2026-10-30. Once done, pull the latest commit of the harness (slurm-agentic-engineering)
  from origin.
- Writing standard (supersedes the ~80% STE rule for owner-facing text): clarity, substance, precision and logical flow;
  about 70% of ASD-STE100 (familiar words, short direct sentences, active voice, consistent terms), with room for natural
  explanation and necessary technical depth; each point builds on the previous one; explain what matters and why; no
  jargon, repetition, vague claims or needless detail; reread as a human reader and fix confusing or wordy parts.
- Until 10:00, the work continues under the existing amendments (5-slot cap, guard, gates).
Execution: a one-shot session cron (c2323022) fires at 10:00 on 2026-10-30 with these steps. Pending jobs are held and
checkpoints are harvested before any cancel; only BC2RL jobs are cancelled.

### Owner amendment A8 (2026-10-29 18:23 EDT) — stop now; one GPU for collation (supersedes the A7 timing)
Faithful record of the owner message: "List the jobs as a list that are running - cancel all except for 1 GPU, and use
that one to collate results and run other checklist tasks. This supersedes the 11AM deadline from earlier - you must stop
at once."
- The A7 wrap-up starts now instead of at 10:00 on 2026-10-30. The 11:00 deadline no longer applies; deliver as soon as
  the collation is done.
- BC2RL keeps one GPU job at most, used for collation and the remaining checklist tasks. No new training.
- The A7 deliverables stay: final report, PROJECT.md stored as a report, STATE/CHECKLIST reset, PROJECT.md skeleton with
  the compute budget, then the harness pull.
Execution (18:24): kept g-dev4 10982233 (running g-yd-s11, the yd evaluation of the Stage G dev checkpoints). Held, then
cancelled r3-c1 10982533 and r3-ctests 10982274 (both pending, never started; nothing to preserve). Deleted crons
84d3eed2 (8-hourly report) and c2323022 (A7 10:00 wrap-up).
- Owner addition (2026-10-29 18:45): for the fresh run, the PROJECT.md skeleton must tell the new agent to ask the owner
  to confirm the GPU limit before any GPU submission: the A8 limit (1 BC2RL GPU job) or the A6 limit (5 preempt slots).
- Owner addition (2026-10-29 18:50): for the harness pull, if `git pull --ff-only` fails, force-pull
  (`git fetch origin && git reset --hard origin/main`). The remote is correct; the local copy may have deviated.
