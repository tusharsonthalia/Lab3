Task B3 — confirmatory seeds, figures, and Part B evidence tables. Fresh session: read predictions.md
Part B (especially B.3, B.6, B.7), results/SCHEMA.md, results/confirm_selection.json and
code/lab3/{prune,cli}.py first. uv only; never weaken tests; do not change frozen choices.

1. Confirmation driver: `uv run python -m lab3 confirm` that reads results/confirm_selection.json and, for
   each of the 12 (arch, method, selected_target) points, runs seeds 1, 2, 3 from the canonical seed-0
   dense parent with the frozen budget (records phase="confirm"). Also run dense_ft with seeds 1, 2, 3 per
   architecture. Total 36 + 9 = 45 arms. Same failure handling and idempotency as the pilot driver.
   Run with nohup and poll with sleep until done.
2. results/confirm_summary.csv: one row per (arch, method, target) with the seed-0 pilot value and the
   seed 1-3 values, mean and sample std of val_acc and test_acc for post_prune and post_finetune; plus
   bytes (dense_zeros, gzip, compact) and latency medians for both paths; plus the dense seed-0 and
   dense_ft (mean over seeds 1-3) reference values.
3. Figures (matplotlib, saved as PNG in figures/, each with axis labels, units, and a caption line in
   figures/CAPTIONS.md that names the source CSV):
   a. figures/acc_vs_sparsity.png — 3 panels (mlp, cnn, gcn); x = achieved eligible-weight sparsity, y =
      test accuracy; one line per method; post_prune dashed, post_finetune solid; horizontal reference
      lines for dense seed-0 and dense_ft; confirmatory points drawn as mean ± std error bars.
   b. figures/latency_vs_accuracy.png — 3 panels; x = latency median (batch 256 for fmnist, full graph
      for Cora; log scale), y = post_finetune test accuracy; marker shape = method, filled = compact path,
      hollow = dense_zeros path; dense baseline marked.
   c. figures/size_vs_sparsity.png — 3 panels; bytes on disk (log) vs achieved sparsity for dense_zeros,
      gzip, and compact artifacts per method; dense baseline line.
4. results/per_layer_sparsity.md: for every pilot arm (post_finetune) a per-layer table (layer, shape,
   numel, nonzero, sparsity), grouped by arch and method; the S2 tables must show the rebuilt shapes.
5. results/failure_accounting.md: every Part B arm/row ever attempted (smoke, pilot, confirm) with status,
   and every failed/excluded record with its classified reason and whether it is excluded from analysis.
   Cells marked "not applicable" must say why (there should be none in the 48-cell matrix — verify).
6. `uv run pytest -q` at the end.

Report: arms ok/failed, the confirm_summary table (compact), figure paths, and any anomaly (e.g. a
confirmed point whose mean val falls below the B.3 threshold, std > 1 pp, compact path faster than dense).
Do not interpret.
