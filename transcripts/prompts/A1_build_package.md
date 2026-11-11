Task A1 — build the dense-reference experiment package for Lab 3.

Read predictions.md section "Part A" first. Everything in A.1 is FROZEN: splits, architectures,
optimizers, epochs, early-stopping rule, seeds, thread counts, latency protocol. Implement it
exactly; if something in it is impossible, stop and say so instead of changing it.

Environment: this is a uv project. Run everything with `uv run ...`; add dependencies only with
`uv add`. CPU only. Do not use pip directly.

Build:
1. A Python package in code/lab3/ with a CLI: `uv run python -m lab3 <cmd> --model {mlp,cnn,gcn} ...`
   commands: `make-splits`, `train`, `eval`, `bench`. Configs in configs/<model>.yaml hold the
   frozen hyperparameters; the CLI reads them (flags may override only for tests).
2. Data: download Fashion-MNIST (torchvision) and Cora (torch_geometric Planetoid, public split)
   into data/ (gitignored, not submitted). `make-splits` writes splits/fmnist_{train,val,test}_idx.npy,
   splits/cora_{train,val,test}_idx.npy and splits/SPLITS.sha256. Every other command loads these
   files and verifies the hash; it must never regenerate splits.
3. Models exactly as in the predictions.md table (no BatchNorm). GCN uses torch_geometric GCNConv.
4. Training with validation-selected checkpointing and the frozen early-stopping rule. Save
   checkpoints/dense/<model>_seed<k>.pt (state_dict + metadata), and per-epoch learning curves
   to results/curves/<run_id>.csv.
5. Logging: append one JSON line per attempted run (including crashes) to results/runs.jsonl
   with a unique run_id (e.g. <model>-<method>-<timestamp>-<shortuuid>) and at least these fields:
   run_id, status (ok/failed + reason), part, architecture, method ("dense"), target_sparsity,
   achieved_sparsity, stored_precision, compute_precision, accumulator_precision, seed,
   train_epochs, finetune_epochs, best_epoch, val_acc, val_loss, test_acc, total_params,
   nonzero_params, dense_bytes, compressed_bytes, est_macs, peak_rss_bytes, latency fields
   (median_ms, iqr_ms, warmup, iters, threads, batch_size or "full_graph"), wall_time_s,
   git-free code hash (sha256 of code/ files), torch/python/numpy versions, timestamp, command.
   Never rewrite or delete existing lines.
6. `bench`: latency per the frozen protocol (inference_mode, eval, 20 warm-up, 200 timed,
   median + IQR, batch 1 and 256 for fmnist, full graph for cora), peak RSS measured in a fresh
   subprocess. MAC estimate: analytical per layer (document the GCN counting convention).
7. Tests in tests/ (pytest, `uv run pytest -q`):
   a. tiny-overfit test per architecture (MLP/CNN: 64 training images to >= 99% train acc;
      GCN: train nodes to >= 99% train acc) — a test may use its own lr/epochs, but must not
      edit configs/;
   b. parameter-count test asserting MLP=235146, CNN=117066, GCN=23063;
   c. evaluation never updates weights: state_dict bit-identical and optimizer untouched after
      `eval` and after `bench`;
   d. split files load, hashes match, splits are disjoint, fmnist val is 500/class.
8. After tests pass: run make-splits, then train dense seed 0 for all three models, then seeds
   1, 2, 3 (replicas), then bench each seed-0 checkpoint. Then write results/baselines.md (a table
   of val/test acc, params, bytes, MACs, latency) and figures/learning_curves.png.

Rules: do not weaken a test to make it pass. If a baseline misses its target (MLP>=82, CNN>=87,
GCN>=65 test acc), stop and report a diagnosis; do not change hyperparameters on your own.
Report at the end: files created, test output, the baseline table, and any anomalies.
