Task C1 — Part C reduced precision: backend support, standalone precision study, and the 12-condition
sparse x INT8 factorial. Fresh session. Read predictions.md Part C (C.1, C.2, C.3) — FROZEN: formats,
calibration rule, INT8 selection rule, sparse operating points, accounting fields. Also read
results/SCHEMA.md, results/partc_selection.json, code/lab3/{bench,models,evaln,compact,prune}.py.
uv only; CPU; frozen threads and latency protocol (20 warm-up, 200 timed, median+IQR, bs1+bs256 for
fmnist, full graph for Cora, RSS in fresh subprocess). Never weaken tests. Never select using test.

1. Backend support report -> results/precision_support.md. Probe on THIS machine and torch 2.15:
   torch.backends.quantized.supported_engines; which quantization APIs exist and work
   (torch.ao.quantization eager quantize_dynamic / prepare+convert, FX prepare_fx/convert_fx, PT2E with
   an available quantizer, torchao if installed — do NOT add torchao unless needed; if you add any
   dependency use `uv add` and record why); whether nn.Linear / Conv2d / GCNConv(.lin) run in fp16 and
   bf16 on CPU (check output dtypes and whether kernels exist or upcast); whether quantized
   linear/conv kernels exist for qnnpack. Every probe result is a row in the report with the evidence
   (command + output). Classify unsupported cases (e.g. unsupported_backend, unsupported_op,
   deprecated_api).
2. Conversions (code/lab3/precision.py), each from the seed-0 dense parent, for all 3 archs:
   fp16, bf16, dynamic INT8, static INT8 (calibrated with splits/fmnist_calib_idx.npy inputs only /
   Cora full-graph features only — record calib size and rule in each row), simulated weight-only INT4
   (per-output-channel symmetric [-8,7], fp32 scales, 2 values packed per uint8 on disk, dequantize to
   fp32 for compute; decide once whether dequantization is inside the timed region, document, apply
   identically; label runtime_status "simulated_non_native").
   For the GCN: if a format cannot be applied to GCNConv, try applying it to the projection (.lin) only
   and record exactly which ops ran in reduced precision; if impossible, write a classified
   unsupported row.
3. Every Part C row (part="C", phase="precision" or "factorial") must contain: stored_precision,
   compute_precision, accumulator_precision (per layer table + summary), val/test acc, actual
   serialized bytes on disk (state of the converted model / packed file), latency, rss,
   frac_macs_reduced_precision (from the per-layer MAC table: MACs of layers that actually executed a
   reduced-precision kernel / total MACs), reload_equivalence {tolerance, max_abs_diff, passed}, and
   parent_run_id. Unsupported conversions still get a row (status "unsupported", classified reason).
4. Tests: reload of each saved converted artifact reproduces logits within the explicit tolerance
   (0 for exact formats; documented tolerance otherwise); INT4 pack/unpack is bit-exact; INT4 values in
   [-8,7]; calibration uses only fmnist_calib_idx inputs (assert ⊂ train split); a test that
   frac_macs_reduced_precision is 0 for fp32 and 1.0 for a model whose every layer ran quantized.
5. Select the best supported INT8 method per architecture by the frozen C.1 rule (validation only; ties
   within one val example -> lower latency); write results/int8_selection.json with the evidence.
6. Factorial (12 conditions): for each arch, dense FP32 (parent), sparse FP32 (the C.3 pilot checkpoint
   from partc_selection.json), dense INT8 (selected method), sparse+INT8 (same method applied to the
   sparse model). For sparse rows report nonzero count before/after INT8 (do zeros survive?), the INT8
   file size, and a sparse-INT8 storage size (int8 values + int32 CSR indices + scales) labelled
   runtime-unsupported unless a real kernel exists. Write results/factorial.csv.
7. Figures: figures/precision_size_latency.png (per arch: bytes on disk vs latency, one point per
   format, unsupported formats listed in the caption) and figures/factorial.png (2x2 per arch:
   accuracy, bytes, latency). One legend each, font >= 8, assert non-empty panels. Captions in
   figures/CAPTIONS.md.
8. results/partc_summary.md: tables of all standalone results and the factorial, raw run_ids included.
9. `uv run pytest -q`.
Report: support matrix, standalone table, the INT8 selection, factorial table, test output, and
anomalies (e.g. reduced-precision slower than fp32, frac_macs < 1 for a "quantized" model). Don't interpret.
