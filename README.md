# Lab 3 — Smaller, Faster Neural Networks with Pi

**Name:** Tushar Sonthalia · **CNetID:** sonthalt

This project compresses three laptop-scale reference models — an MLP and a CNN on
Fashion-MNIST, and a two-layer GCN on Cora — with four sparsity methods (S1–S4) and
several reduced-precision formats. Every claim is backed by measured accuracy, on-disk
bytes, memory, and warmed CPU latency.

## Hardware / software

| | |
|---|---|
| Machine | Apple M4 Pro (Mac16,8), 12 CPU cores, 24 GB RAM, macOS 27.0 |
| Device | **CPU only** for every arm; `torch.set_num_threads(4)`, interop threads 1 |
| Python | 3.12.13 (pinned: `.python-version`, `requires-python >=3.12,<3.13`) |
| Libraries | PyTorch 2.15.0, torchvision 0.29.2, torch_geometric 2.8.1, NumPy 2.5.4 (locked in `uv.lock`) |
| Quantized engine | `qnnpack` (the only engine available on this machine) |
| Coding agent | Pi 1.1.0 on the course endpoint `dls-qwen` / `qwen3.8-27b` |

**Total compute time.**
- Sum of `wall_time_s` over all records in `results/runs.jsonl`: about **60 min**
  (Part A 11.7 min, Part B 44.6 min, Part C 3.8 min).
- Pi model time across all sessions: about **6.6 h**.

## Setup

```bash
uv sync          # creates .venv from uv.lock (Python 3.12)
export PYTHONPATH=code
uv run python -m lab3 make-splits   # downloads Fashion-MNIST + Cora into data/, verifies the saved split hashes
uv run pytest -q                    # 153 tests
```

`PYTHONPATH=code` is required. `uv` writes its editable-install `.pth` files with the macOS
`UF_HIDDEN` flag, and Python 3.12.13's `site` module skips hidden `.pth` files. Without the
variable, `import lab3` fails. (pytest sets this itself via `pythonpath = ["code"]`.)

The saved splits in `splits/` are used as-is. `make-splits` verifies them against
`splits/SPLITS.sha256` and never regenerates them.

## Replay — one command per part

All commands assume `PYTHONPATH=code` and are run from the project root.

**Part A — dense baselines (seed 0 = canonical parent; seeds 1–3 = replicas):**
```bash
for m in mlp cnn gcn; do for s in 0 1 2 3; do uv run python -m lab3 train --model $m --seed $s; done; uv run python -m lab3 bench --model $m --seed 0; done
```

**Part B — 48-arm pilot + dense_ft controls → validation-only selection → 3-seed confirmation → figures:**
```bash
uv run python -m lab3 make-calib && uv run python -m lab3 pilot && uv run python scripts/select_confirm.py && uv run python -m lab3 confirm && uv run python scripts/make_figures.py && uv run python scripts/make_per_layer_md.py && uv run python scripts/make_failure_accounting.py
```

**Part C — precision study → operating-point selection → 12-condition factorial → summary:**
```bash
uv run python scripts/select_partc.py && uv run python -m lab3 precision && uv run python -m lab3 factorial && uv run python -m lab3 c2summary && uv run python scripts/make_figures.py
```

**Provenance check of `predictions.md` against the transcripts:**
```bash
uv run python scripts/predictions_provenance.py
```

## Where things are

| Path | Contents |
|---|---|
| `predictions.md` | Dated, append-only predictions and frozen design. Timing evidence: `results/predictions_provenance.md` |
| `code/lab3/`, `tests/` | Package and tests (implemented by Pi) |
| `configs/`, `splits/` | Frozen hyperparameters; saved train/val/test/calibration indices + hashes |
| `results/runs.jsonl` | One immutable record per attempted run, failures included. Exclusions and corrections: `results/ANNOTATIONS.md`; fields: `results/SCHEMA.md` |
| `results/` summaries | `baselines.md`, `pilot_summary.csv`, `confirm_summary.csv`, `per_layer_sparsity.md`, `failure_accounting.md`, `precision_support.md`, `partc_summary.md`, `factorial.csv`, selection JSONs |
| `checkpoints/` | Dense baselines (`dense/`) and the selected compressed exports |
| `figures/` | Trade-off plots; captions in `figures/CAPTIONS.md` |
| `transcripts/` | Complete Pi transcripts (`*.jsonl`, `pi-sessions/`) and every prompt sent (`prompts/`) |
| `AGENT_DISCLOSURE.md` | What Pi did |
| `MANIFEST.sha256` | Hashes of all submitted files |

## Reading the records

**Superseded rows.** Some rows in `runs.jsonl` were replaced by later reruns. Each
replacement is listed in `results/ANNOTATIONS.md` with its reason. Examples:
- debug and smoke runs;
- the first Part C run, which used the wrong Python version;
- INT4, artifact-format, and accounting bugs.

The rows are never deleted. Summaries and figures use the latest non-superseded record.

**Measurement notes.**
- **Latency:** after 20 warm-up iterations, 200 timed iterations; median and IQR are
  reported. Batch 1 and 256 for Fashion-MNIST, one full-graph forward for Cora.
- **Memory:** RSS measured in a fresh subprocess.
- **Bytes:** actual file sizes from `os.path.getsize`.
- **INT4 is simulated.** The weights are stored packed at 4 bits, then dequantized once and
  cached, and compute runs in FP32. The dequantization cost is outside the timed region;
  see `ANNOTATIONS.md`.
