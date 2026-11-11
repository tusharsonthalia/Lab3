# Agent Disclosure

**Coding agent:** Pi 1.1.0, using the course endpoint `dls-qwen` / `qwen3.8-27b`. It ran
non-interactively (`pi -p --mode json`), one session per task, under `uv`. The complete,
unedited transcripts are in `transcripts/` (`*.jsonl` streams and `pi-sessions/`). The
exact prompt for every session is in `transcripts/prompts/`.

## What Pi implemented

- **Package `code/lab3/`:**
  - CLI: `make-splits`, `make-calib`, `train`, `eval`, `bench`, `prune`, `pilot`,
    `confirm`, `precision`, `factorial`, `c2summary`.
  - Data loading and deterministic splits, plus the three reference models.
  - Training with validation-selected checkpointing.
  - Benchmarking: latency, RSS, and MACs computed from layer shapes.
  - The four sparsity methods: S1–S4 masks and S2 physical surgery, CSR / N:M / rebuilt
    compact exports, and the reversible mapping.
  - Fine-tuning with mask re-application, and the pilot and confirmation drivers.
  - Reduced-precision conversions: FP16/BF16, native qnnpack dynamic and FX static INT8,
    and simulated INT4.
  - The sparse × INT8 factorial.
- **Tests in `tests/`:** overfit, parameter count, no-update, splits, sparsity invariants,
  reload equivalence, compact/dense logit equivalence, and precision/accounting tests.
- **Experiment execution:** all records in `results/runs.jsonl`, including failed and
  debug attempts. Excluded rows are listed in `results/ANNOTATIONS.md`.
- **Summaries, figures, and their generation scripts:** `scripts/make_figures.py`,
  `scripts/make_per_layer_md.py`, and `scripts/make_failure_accounting.py`.

## What was specified to Pi (not decided by Pi)

These were all frozen in `predictions.md` before the governing runs and passed to Pi in
the prompts:
- Splits, seeds, and architectures.
- Epoch budgets, the early-stopping rule, and the fine-tuning budget.
- Pruning levels and N:M patterns, the S2 group definitions, and the S3 score.
- The calibration rule and the validation-only selection rules.
- The latency protocol and the required record fields.

Corrections to Pi's work after review are stated in the follow-up prompts. Examples: MAC
conventions, CSR orientation, native INT8 instead of emulation, and the INT4 and
artifact-content bugs.