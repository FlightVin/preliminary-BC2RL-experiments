# BC2RL — initial success profiles and online learning

This repository holds the BC2RL first arithmetic study: behaviour cloning (BC) initialisations of
`Qwen/Qwen2.5-0.5B-Instruct`, then GRPO or online filtered supervision, evaluated with pass@1,
pass@4 and pass^4.

- Protocol: `protocol/PROJECT.md` (owner specification) and
  `protocol/rl_initialization_research_proposal.pdf` (the proposal it narrows).
- CLI: `python -m bc2rl {generate,bc,online,evaluate,report} --config <yaml>`.
- Environment: `scripts/setup_env.sh` builds the venv under `$PROJ_DATA` on a compute node.
